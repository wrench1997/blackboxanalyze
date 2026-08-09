"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./pg385-demo.module.css";

type Stage = 0 | 1 | 2 | 3 | 4 | 5;
type Method = "GET" | "POST";
type Implementation = "Python A" | "Node B";

const stages = [
  { eyebrow: "01", title: "发送基线探针", detail: "受控 canary 进入本地过滤器" },
  { eyebrow: "02", title: "读取脱敏反馈", detail: "模型只看到 abstract filter projection" },
  { eyebrow: "03", title: "组装 Rule-IR", detail: "选择 one-variable encoding repair" },
  { eyebrow: "04", title: "绑定并发送", detail: "reviewed adapter 在最后一跳生成字符串" },
  { eyebrow: "05", title: "C / R / N / replay", detail: "candidate、reference、negative、fresh replay" },
];

const slotRows = [
  ["next_action", "repair", "动作改变"],
  ["repair_action", "encoding", "只改编码层"],
  ["encoding_ref", "double_layer_order_sensitive", "变体引用"],
  ["probe_variant_ref", "one_variable_repair", "受控变体"],
  ["safe_to_send", "1", "由 evaluator 再校验"],
  ["oracle_ref", "response_shape", "脱敏 typed oracle"],
];

const axisRows = [
  ["transport", "GET / POST", "保留方法与 placement"],
  ["parameter", "query_term / form_field", "不丢参数角色"],
  ["encoding", "identity → double layer", "失败反馈驱动"],
  ["response", "bounded projection", "正文不进上下文"],
  ["history", "baseline → repair → replay", "保留动作轨迹"],
  ["negative", "matched triplet", "负对照零误放"],
];

const filterRows = [
  ["raw delimiter", "PG386_<ROLE>_0002:", "filtered", "raw_delimiter_blocked"],
  ["single encoding", "PG386_<ROLE>_0002%3A", "filtered", "encoding_filter"],
  ["double encoding", "PG386_<ROLE>_0002%25253A", "typed effect", "encoded_variant"],
  ["matched negative", "PG386_NEG_0002%25253A", "no effect", "negative_control"],
];

const ctfContextRows = [
  { label: "DOM text reflection", sink: "dom_text", sinkContext: "text_sink", source: "location_search", parser: "url_search_params", filter: "none_observed", guard: "none_observed", control: "straight_line", loader: "static_only", state: "ephemeral", normalization: "search_params_then_decode", action: "select_probe_variant", note: "纯文本落点；不会把输入当 HTML 执行" },
  { label: "attribute reflection", sink: "dom_attribute", sinkContext: "attribute_sink", source: "location_search", parser: "url_search_params", filter: "allowlist_or_membership", guard: "conditional", control: "branch", loader: "static_only", state: "ephemeral", normalization: "attribute_escape_then_render", action: "ask", note: "需要确认属性上下文和编码顺序" },
  { label: "HTML fragment guard", sink: "dom_html_guarded", sinkContext: "html_fragment_sink", source: "form_input", parser: "form_decode", filter: "sanitizer_or_escape", guard: "policy_gate", control: "branch", loader: "dynamic_blocked", state: "ephemeral", normalization: "sanitize_then_parse", action: "abstain", note: "动态 loader/HTML 执行门未满足" },
  { label: "JSON parser boundary", sink: "json_value", sinkContext: "sink_not_observed", source: "form_input", parser: "json_parse", filter: "allowlist_or_membership", guard: "conditional", control: "branch+exception", loader: "static_only", state: "ephemeral", normalization: "json_parse_then_validate", action: "ask", note: "先区分解析失败和业务效果" },
  { label: "double-decode order", sink: "dom_text", sinkContext: "text_sink", source: "location_search", parser: "url_search_params", filter: "blocklist_or_regex", guard: "conditional", control: "branch", loader: "static_only", state: "ephemeral", normalization: "double_decode_order_sensitive", action: "repair", note: "只改变 encoding，不改变 sink/transport" },
  { label: "persistent state guard", sink: "dom_text", sinkContext: "text_sink", source: "form_input", parser: "form_decode", filter: "policy_gate", guard: "policy_gate", control: "branch", loader: "static_only", state: "persistent_blocked", normalization: "search_params_then_decode", action: "abstain", note: "持久化写入不在演示权限内" },
];

const implementations = [
  { name: "Python A", runtime: "threading loopback", route: "/pg385/filter", field: "q", source: "49f2c63c…", color: "blue" },
  { name: "Node B", runtime: "native HTTP", route: "/pg385b/filter", field: "value", source: "ede5e6e7…", color: "orange" },
];

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function statusFor(stage: Stage, index: number) {
  if (stage === 5 || index < stage) return "done";
  if (index === stage - 1) return "active";
  return "idle";
}

export default function Pg385Demo() {
  const [stage, setStage] = useState<Stage>(0);
  const [running, setRunning] = useState(false);
  const [method, setMethod] = useState<Method>("GET");
  const [implementation, setImplementation] = useState<Implementation>("Python A");
  const [showWire, setShowWire] = useState(false);
  const [showGeneratedValue, setShowGeneratedValue] = useState(false);
  const [filterTraceIndex, setFilterTraceIndex] = useState(-1);
  const [filterTraceRunning, setFilterTraceRunning] = useState(false);
  const [ctfContextIndex, setCtfContextIndex] = useState(0);
  const runToken = useRef(0);
  const filterTraceToken = useRef(0);

  useEffect(() => () => { runToken.current += 1; filterTraceToken.current += 1; }, []);

  async function runDemo() {
    if (running) return;
    const token = runToken.current + 1;
    runToken.current = token;
    setRunning(true);
    setShowWire(false);
    setShowGeneratedValue(false);
    setStage(1);
    await wait(620);
    if (runToken.current !== token) return;
    setStage(2);
    await wait(720);
    if (runToken.current !== token) return;
    setStage(3);
    await wait(720);
    if (runToken.current !== token) return;
    setStage(4);
    await wait(720);
    if (runToken.current !== token) return;
    setStage(5);
    setRunning(false);
  }

  async function runFilterTrace() {
    if (filterTraceRunning) return;
    const token = filterTraceToken.current + 1;
    filterTraceToken.current = token;
    setFilterTraceRunning(true);
    setFilterTraceIndex(-1);
    for (let index = 0; index < filterRows.length; index += 1) {
      await wait(520);
      if (filterTraceToken.current !== token) return;
      setFilterTraceIndex(index);
    }
    setFilterTraceRunning(false);
  }

  const selected = implementations.find((item) => item.name === implementation) || implementations[0];
  const wire = method === "GET"
    ? `GET http://127.0.0.1:<ephemeral-port>${selected.route}?${selected.field}=PG385_CAND_0002%25253A`
    : `POST http://127.0.0.1:<ephemeral-port>${selected.route}\nContent-Type: application/x-www-form-urlencoded\n\n${selected.field}=PG385_CAND_0002%25253A`;
  const generatedValue = "PG386_CAND_0002%25253A";
  const generatedWire = method === "GET"
    ? `model → ${selected.field}=${generatedValue}`
    : `model → ${selected.field}=${generatedValue}`;
  const ctfContext = ctfContextRows[ctfContextIndex];

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <a className={styles.brand} href="/" aria-label="返回 SIFT 研究台"><span>S</span><b>SIFT</b><small>PG-385 / FILTER LAB</small></a>
        <div className={styles.headerLinks}><a href="/">主研究台</a><a href="/ops">运营证据</a><span className={styles.liveDot}><i />LOOPBACK ONLY</span></div>
      </header>

      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <p className={styles.kicker}>MODEL-SELECTED REPAIR / TWO IMPLEMENTATIONS</p>
          <h1>让过滤反馈，<em>推动下一步。</em></h1>
          <p className={styles.lead}>一个被过滤的测试字符串，不直接喂给模型。模型读取脱敏反馈，选择抽象变体；受审阅 evaluator 再把变体绑定到本地 canary，完成 typed 复放。</p>
          <div className={styles.heroChips}><span className={styles.chipAccent}>MODEL SELECTED</span><span>GET + POST</span><span>C / R / N / REPLAY</span><span>PROMOTION CLOSED</span></div>
        </div>
        <div className={styles.heroCard}>
          <div className={styles.cardOverline}><span>PG-385 / CANDIDATE</span><b>READY TO SHOW</b></div>
          <div className={styles.heroMetric}><strong>4 / 4</strong><span>model-selected rows</span></div>
          <div className={styles.heroMetricGrid}><div><strong>2</strong><span>implementations</span></div><div><strong>0</strong><span>negative false allow</span></div><div><strong>1.0</strong><span>encoding exact</span></div><div><strong>0</strong><span>raw in context</span></div></div>
          <p className={styles.cardNote}>A800 selector · seed 38503 · abstract checkpoint</p>
        </div>
      </section>

      <section className={styles.loopSection}>
        <div className={styles.sectionHead}><div><p className={styles.kicker}>01 · LIVE WALKTHROUGH</p><h2>从被拒绝，到一变量修复。</h2></div><p>这是演示状态机，不是任意目标攻击器。每次运行都在固定本地实现中 fresh reset。</p></div>
        <div className={styles.controlBar}>
          <label>实现<select value={implementation} onChange={(event) => setImplementation(event.target.value as Implementation)} disabled={running}><option>Python A</option><option>Node B</option></select></label>
          <label>方法<select value={method} onChange={(event) => setMethod(event.target.value as Method)} disabled={running}><option>GET</option><option>POST</option></select></label>
          <div className={styles.controlSummary}><span>route</span><b>{selected.route}</b><span>field</span><b>{selected.field}</b></div>
          <button className={styles.primaryButton} onClick={runDemo} disabled={running}>{running ? "运行中…" : stage === 5 ? "再跑一遍" : "运行模型闭环"}<span>↗</span></button>
        </div>

        <div className={styles.loopGrid}>
          <div className={styles.timeline}>
            {stages.map((item, index) => {
              const status = statusFor(stage, index + 1);
              return <div className={`${styles.timelineItem} ${styles[status]}`} key={item.eyebrow}><div className={styles.stepNumber}>{status === "done" ? "✓" : item.eyebrow}</div><div><b>{item.title}</b><p>{item.detail}</p></div><span className={styles.stepState}>{status === "done" ? "PASS" : status === "active" ? "LIVE" : "WAIT"}</span></div>;
            })}
            <div className={styles.timelineFoot}><span className={styles.statusPill}><i />{stage === 5 ? "TYPED REPLAY COMPLETE" : running ? "LOCAL EVALUATOR RUNNING" : "READY"}</span><span>fresh reset per role</span></div>
          </div>
          <div className={styles.stagePanel}>
            <div className={styles.panelLabel}><span>MODEL / EVALUATOR SPLIT</span><b>{stage === 5 ? "EVIDENCE" : "ABSTRACT"}</b></div>
            {stage < 2 && <div className={styles.emptyState}><span className={styles.emptyGlyph}>→</span><h3>先发一个基线</h3><p>点击“运行模型闭环”，观察过滤反馈如何进入下一步，而不是进入原始字符串。</p></div>}
            {stage >= 2 && <div className={styles.feedbackBlock}><div className={styles.feedbackHeader}><span>FILTER FEEDBACK / SANITIZED</span><strong>filtered</strong></div><div className={styles.feedbackTokens}><code>state=filtered</code><code>class=encoding_filter</code><code>failure=raw_delimiter_blocked</code><code>acceptance=encoded_variant_required</code></div><div className={styles.feedbackArrow}>↓ <span>model reads only these tokens</span></div><div className={styles.ruleBox}><div className={styles.ruleHeader}><span>MODEL-SELECTED RULE-IR</span><b>safe abstract slots</b></div><div className={styles.slotGrid}>{slotRows.map(([key, value, note]) => <div key={key}><code>{key}</code><strong>{value}</strong><span>{note}</span></div>)}</div></div></div>}
            {stage === 5 && <div className={styles.resultStrip}><div><strong>1 / 1</strong><span>candidate typed</span></div><div><strong>1 / 1</strong><span>reference typed</span></div><div className={styles.negative}><strong>0</strong><span>negative violation</span></div><div><strong>1 / 1</strong><span>replay typed</span></div></div>}
          </div>
        </div>

        {stage === 5 && <div className={styles.wirePanel}><div><div className={styles.panelLabel}><span>EPHEMERAL LOCAL WIRE PREVIEW</span><b>NOT PERSISTED</b></div><p>这是最后一跳的可见请求形状。端口是临时的，字符串只存在于 evaluator 进程内；报告只保存 hash、投影和证据。</p></div><button className={styles.secondaryButton} onClick={() => setShowWire((value) => !value)}>{showWire ? "隐藏临时 wire" : "显示临时 wire"}</button>{showWire && <pre>{wire}</pre>}</div>}
      </section>

      <section className={styles.payloadSection}>
        <div className={styles.sectionHead}><div><p className={styles.kicker}>01A · MODEL-GENERATED FIXTURE VALUE</p><h2>模型真的能生成一条受约束的测试字符串。</h2></div><p>PG-386 在 PG-385 token backbone 上增加 grammar head。只允许预注册本地过滤器的 canary 形状，越界就不发送。</p></div>
        <div className={styles.payloadGrid}>
          <article className={styles.payloadCard}><span>MODEL OUTPUT</span><strong>{showGeneratedValue ? generatedValue : "PG386_(CAND|REF|NEG|REPLAY)_0002%25253A"}</strong><p>角色、方法和过滤反馈来自抽象 token；具体值只在 loopback evaluator 里短暂展开。</p><button className={styles.secondaryButton} onClick={() => setShowGeneratedValue((value) => !value)}>{showGeneratedValue ? "隐藏字符串" : "显示临时字符串"}</button></article>
          <article className={styles.payloadCard}><span>BOUNDARY CHECK</span><div className={styles.payloadChecks}><b>grammar valid</b><b>local only</b><b>GET + POST</b><b>raw not persisted</b></div><p>{generatedWire}</p></article>
          <article className={styles.payloadCard}><span>RESULT</span><strong>4 / 4 typed</strong><p>Python A / Node B，candidate、reference、replay 均复放；negative violation=0。</p><span className={styles.statusPill}><i />PROMOTION CLOSED</span></article>
        </div>
      </section>

      <section className={styles.algorithmSection}>
        <div className={styles.sectionHead}><div><p className={styles.kicker}>01B · FILTER / JUDGMENT TRACE</p><h2>过滤器、模型和 oracle 各自做什么。</h2></div><p>下面是 PG-385/386 本地 fixture 的真实抽象逻辑。它不是把任意 WAF 规则猜成通用绕过，而是验证一次可复现的 canonicalization 差分。</p></div>
        <div className={styles.algorithmGrid}>
          <article className={styles.algorithmCard}><div className={styles.panelLabel}><span>FILTER ALGORITHM</span><b>LOCAL FIXTURE</b></div><pre>{`source = GET query or POST form field\nif raw contains ":" or single "%3A":\n    return filtered / 4xx\nnormalized = decode_percent(source, max_layers=3)\nif ":" not in normalized:\n    return no_effect\nif "_NEG_" in normalized:\n    return matched_negative / no_effect\nreturn typed_effect / bounded_marker_reflection`}</pre></article>
          <article className={styles.algorithmCard}><div className={styles.panelLabel}><span>DECISION TABLE</span><b>FEEDBACK → OUTCOME</b></div><div className={styles.filterTable}><div className={styles.filterTableHead}><span>INPUT SHAPE</span><span>FIXTURE VALUE</span><span>OUTCOME</span><span>ABSTRACT REASON</span></div>{filterRows.map(([label, value, state, reason], index) => <div className={`${styles.filterTableRow} ${filterTraceIndex === index ? styles.filterTableRowActive : ""}`} key={label}><code>{label}</code><strong>{value}</strong><span className={state === "typed effect" ? styles.goodState : state === "no effect" ? styles.neutralState : styles.badState}>{state}</span><small>{reason}</small></div>)}</div><button className={styles.traceButton} onClick={runFilterTrace} disabled={filterTraceRunning}>{filterTraceRunning ? `过滤器逐步判断中… ${Math.max(filterTraceIndex + 1, 0)} / ${filterRows.length}` : filterTraceIndex === filterRows.length - 1 ? "重新跑一遍过滤判断" : "逐步运行过滤判断"}<span>↗</span></button></article>
          <article className={styles.algorithmCard}><div className={styles.panelLabel}><span>MODEL JUDGMENT</span><b>ABSTRACT TOKENS</b></div><div className={styles.judgmentStack}><div><code>filter_state=filtered</code><code>failure_shape=raw_delimiter_blocked</code><code>encoding_acceptance=encoded_variant_required</code></div><span>↓</span><div><code>encoding_ref=double_layer_order_sensitive</code><code>probe_variant_ref=one_variable_repair</code><code>repair_action=encoding</code><code>next_action=repair</code><code>safe_to_send=1</code></div><p>{filterTraceIndex >= 0 ? `当前步骤：${filterRows[filterTraceIndex][0]} → ${filterRows[filterTraceIndex][2]}；模型只更新 encoding_ref，其他槽位保持不变。` : "点击左侧逐步运行，先看到过滤器的结果，再看到模型如何做一变量修复。"}</p></div></article>
        </div>
        <div className={styles.algorithmFoot}><span>绕过判定 = 过滤器先拒绝 → 第二编码层通过 → bounded marker typed effect</span><span>不是脚本执行、SQL 执行、持久化或外连</span></div>
      </section>

      <section className={styles.ctfSection}>
        <div className={styles.sectionHead}><div><p className={styles.kicker}>01C · CTF FRONTEND CONTEXTS</p><h2>先读页面脚本，再决定要不要动。</h2></div><p>这些是本地 CTF-like 语境模板，不是外部站点答案。模型看到的是 sink、loader、状态、规范化和响应形状 token；源码只在 evaluator 侧做一次投影。</p></div>
        <div className={styles.ctfGrid}>
          <div className={styles.ctfCaseList}>
            <div className={styles.panelLabel}><span>CONTEXT CASES</span><b>16 ABSTRACT CLASSES</b></div>
            {ctfContextRows.map((item, index) => <button className={`${styles.ctfCase} ${ctfContextIndex === index ? styles.ctfCaseActive : ""}`} key={item.label} onClick={() => setCtfContextIndex(index)}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item.label}</strong><small>{item.sink} · {item.action}</small></button>)}
          </div>
          <div className={styles.ctfReadout}>
            <div className={styles.panelLabel}><span>JS CONTEXT PROJECTION</span><b>NO SOURCE STORED</b></div>
            <pre>{`const value = query.get("q");
preview.${ctfContext.sink === "dom_attribute" ? "setAttribute(name, value)" : ctfContext.sink === "dom_text" ? "textContent = value" : "guardedRender(value)"};`}</pre>
            <div className={styles.ctfTokens}><code>js_source={ctfContext.source}</code><code>js_parser={ctfContext.parser}</code><code>js_sink_context={ctfContext.sinkContext}</code><code>js_filter_shape={ctfContext.filter}</code><code>js_guard_shape={ctfContext.guard}</code><code>js_control_flow={ctfContext.control}</code><code>loader_policy={ctfContext.loader}</code><code>state_policy={ctfContext.state}</code><code>normalization={ctfContext.normalization}</code></div>
            <div className={styles.ctfDecision}><span>MODEL DECISION</span><strong>{ctfContext.action}</strong><p>{ctfContext.note}</p></div>
            <div className={styles.ctfGate}><b>{ctfContext.action === "abstain" ? "BLOCKED / ASK" : ctfContext.action === "ask" ? "ASK FOR MISSING CONTEXT" : ctfContext.action === "repair" ? "ONE-VARIABLE REPAIR" : "CONTROLLED PROBE"}</b><span>external loader=false · persistent write=false · raw source=false</span></div>
          </div>
        </div>
      </section>

      <section className={styles.matrixSection}>
        <div className={styles.sectionHead}><div><p className={styles.kicker}>02 · CROSS-IMPLEMENTATION EVIDENCE</p><h2>同一抽象决策，换实现再复放。</h2></div><p>模型不读取 implementation name、route literal 或响应正文。只有 evaluator 知道本地绑定。</p></div>
        <div className={styles.matrix}>
          <div className={styles.matrixHead}><span>IMPLEMENTATION</span><span>RUNTIME / ROUTE</span><span>GET</span><span>POST</span><span>NEGATIVE</span><span>REPLAY</span></div>
          {implementations.map((item) => <div className={styles.matrixRow} key={item.name}><div className={styles.implName}><i className={item.color === "blue" ? styles.blueDot : styles.orangeDot} /><strong>{item.name}</strong><small>{item.source}</small></div><div><b>{item.runtime}</b><small>{item.route} · {item.field}</small></div><span className={styles.pass}>1 / 1</span><span className={styles.pass}>1 / 1</span><span className={styles.pass}>0</span><span className={styles.pass}>1 / 1</span></div>)}
        </div>
        <div className={styles.provenance}><span className={styles.statusPill}><i />SOURCE HASH BOUND</span><span className={styles.statusPill}><i />LOOPBACK ONLY</span><span className={styles.statusPill}><i />NO BUSINESS WRITE</span><span className={styles.statusPill}><i />PROMOTION FALSE</span></div>
      </section>

      <section className={styles.informationSection}>
        <div className={styles.sectionHead}><div><p className={styles.kicker}>03 · INFORMATION PRESERVATION</p><h2>不为了好看，压掉页面信息。</h2></div><p>token 轴保留方法、角色、编码、响应形状、动作历史和负对照；熵门只在首阶段 next-token 上硬阻断，后续 adapter 只做诊断。</p></div>
        <div className={styles.axisGrid}>{axisRows.map(([axis, value, note]) => <article key={axis}><span>{axis}</span><strong>{value}</strong><p>{note}</p></article>)}</div>
        <div className={styles.entropyBar}><div><span>STAGE 1 / NEXT TOKEN</span><strong>≤ 25% entropy drop</strong><p>硬门：信息塌缩就隔离候选。</p></div><div className={styles.entropyArrow}>→</div><div><span>STAGE 2+ / ADAPTER & REPAIR</span><strong>diagnostic, not sole blocker</strong><p>仍然检查 finite logits、slot coverage、ASK、negative 和 holdout leakage。</p></div></div>
      </section>

      <section className={styles.guardSection}>
        <div className={styles.sectionHead}><div><p className={styles.kicker}>04 · CLAIM BOUNDARY</p><h2>看到什么，能说到哪里。</h2></div><p>把“模型选了变体”和“漏洞已经在任意网址复现”分开，展示可信度反而更高。</p></div>
        <div className={styles.guardGrid}><article className={styles.guardGood}><span>CAN SHOW</span><h3>抽象闭环</h3><ul><li>缺观测 → ASK</li><li>过滤反馈 → Rule‑IR 修复</li><li>受审阅本地 canary → typed effect</li><li>candidate / reference / negative / replay</li></ul></article><article className={styles.guardHold}><span>STILL CLOSED</span><h3>不能宣称</h3><ul><li>任意网址 WAF 绕过</li><li>持久化 XSS 或反链</li><li>任意原始攻击字符串生成</li><li>payload catalog / memory promotion</li></ul></article><article className={styles.guardInfo}><span>ARTIFACTS</span><h3>证据留什么</h3><ul><li>abstract slots + model hash</li><li>template / source hash</li><li>bounded response projection</li><li>evidence SHA-256 + fresh reset</li></ul></article></div>
      </section>

      <footer className={styles.footer}><a className={styles.brand} href="/"><span>S</span><b>SIFT</b></a><p>Small model. Long memory. Executable evidence.</p><a href="/">返回主研究台 ↑</a></footer>
    </main>
  );
}
