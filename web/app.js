const state = {
  scenarios: [],
  session: null,
  candidates: [],
  closure: null,
  activeQueries: [],
  running: false,
  protocolStep: 0,
};

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const closureLabels = {
  insufficient_data: "数据不足",
  open: "假设空间仍开放",
  identified: "规则已识别",
  observationally_closed: "观测行为闭环",
  observationally_closed_low_coverage: "低覆盖闭环",
  deadlocked: "确认状态死路",
  suspected_deadlock: "疑似状态死路",
  context_incomplete_or_nondeterministic: "发现隐藏上下文",
  dsl_or_search_insufficient: "规则语言不足",
  budget_or_domain_limited: "实验预算受限",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function toast(message, type = "info") {
  const item = document.createElement("div");
  item.className = `toast ${type === "error" ? "error" : ""}`;
  item.textContent = message;
  $("toast-region").append(item);
  setTimeout(() => item.remove(), 3600);
}

function selectedScenario() {
  return state.scenarios.find((item) => item.id === $("scenario-select").value) || state.scenarios[0];
}

function setPath(root, path, value) {
  const parts = path.split(".");
  let cursor = root;
  for (const part of parts.slice(0, -1)) cursor = cursor[part] ||= {};
  cursor[parts.at(-1)] = value;
}

function parseOptionalJson(text, label) {
  if (!text.trim()) return undefined;
  try { return JSON.parse(text); }
  catch (error) { throw new Error(`${label} 不是合法 JSON：${error.message}`); }
}

function envelopeFromForm() {
  const envelope = { input: {}, context: {}, state: {} };
  document.querySelectorAll("[data-field-path]").forEach((element) => {
    const type = element.dataset.fieldType;
    const value = type === "bool" ? element.checked : type === "number" ? Number(element.value) : element.value;
    setPath(envelope, element.dataset.fieldPath, value);
  });
  envelope.episode_id = $("episode-id").value.trim() || "default";
  if ($("episode-step").value !== "") envelope.step = Number($("episode-step").value);
  if ($("goal-reached").checked) envelope.goal = true;
  const stateAfter = parseOptionalJson($("state-after-json").value, "state_after");
  if (stateAfter !== undefined) envelope.state_after = stateAfter;
  return envelope;
}

function fillForm(envelope) {
  document.querySelectorAll("[data-field-path]").forEach((element) => {
    let value = envelope;
    for (const part of element.dataset.fieldPath.split(".")) value = value?.[part];
    if (value === undefined) return;
    if (element.dataset.fieldType === "bool") element.checked = Boolean(value);
    else element.value = String(value);
  });
}

function renderFields(fields = []) {
  const target = $("field-form");
  target.innerHTML = "";
  for (const spec of fields) {
    const label = document.createElement("label");
    label.append(document.createTextNode(`${spec.label || spec.path} · ${spec.path}`));
    let input;
    if (spec.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(spec.domain?.[0]);
    } else if (Array.isArray(spec.domain) && spec.domain.length <= 30) {
      input = document.createElement("select");
      for (const value of spec.domain) {
        const option = document.createElement("option");
        option.value = String(value);
        option.textContent = String(value);
        input.append(option);
      }
    } else {
      input = document.createElement("input");
      input.type = spec.type === "number" ? "number" : "text";
      input.value = spec.domain?.[0] ?? "";
    }
    input.dataset.fieldPath = spec.path;
    input.dataset.fieldType = spec.type;
    label.append(input);
    target.append(label);
  }
}

function renderScenario() {
  const scenario = selectedScenario();
  if (!scenario) return;
  $("scenario-name").textContent = scenario.name;
  $("scenario-description").textContent = scenario.description || "";
  $("scenario-question").textContent = scenario.research_question || "等待定义研究问题。";
  $("scenario-hypothesis").textContent = scenario.hypothesis || "等待定义可证伪假设。";
  $("scenario-category").textContent = scenario.category || "通用规则归纳";
  $("scenario-cwe").textContent = scenario.cwe || "BENCHMARK";
  $("scenario-severity").textContent = scenario.severity || "BASELINE";
  $("scenario-severity").className = `severity ${(scenario.severity || "").toLowerCase()}`;
  $("metric-cwe").textContent = `${scenario.cwe || "—"} · ${scenario.category || "规则研究"}`;
  $("js-source").textContent = scenario.js_source || "// 此基线场景没有附带 JavaScript 语料。";
  $("intended-rule").textContent = scenario.intended_rule || "由研究者定义预期语义";
  $("game-rule").textContent = scenario.game_rule || scenario.description || "等待语义抽象。";
  $("scenario-tags").innerHTML = (scenario.tags || ["rule-induction"]).map((tag) => `<span>#${escapeHtml(tag)}</span>`).join("");
  if (!state.running) {
    $("metric-risk").textContent = scenario.severity || "基线";
    $("metric-confidence").textContent = "—";
  }
}

function setProtocolStep(step, message) {
  state.protocolStep = step;
  $("protocol-progress").style.width = `${Math.min(100, step * 25)}%`;
  $("protocol-progress-label").textContent = `${step} / 4`;
  if (message) $("run-message").textContent = message;
  document.querySelectorAll(".pipeline-step").forEach((element, index) => {
    element.classList.toggle("done", index < step - 1 || (step === 4 && !state.running));
    element.classList.toggle("active", state.running && index === step - 1);
  });
  const stageIds = ["sieve-observe", "sieve-induce", "sieve-stress", "sieve-verdict"];
  stageIds.forEach((id, index) => {
    $(id).classList.toggle("complete", index < step - 1 || (step === 4 && !state.running));
    $(id).classList.toggle("active", state.running && index === step - 1);
  });
}

function setRunStatus(status, label) {
  $("run-status").className = `run-status ${status}`;
  $("run-status").innerHTML = `<i></i>${escapeHtml(label)}`;
}

function renderMetrics() {
  const observations = state.session?.observations || [];
  $("metric-observations").textContent = observations.length;
  $("metric-budget").textContent = `预算 ${observations.length} / 18 queries`;
  $("metric-candidates").textContent = state.candidates.length;
  $("metric-accuracy").textContent = state.candidates[0] ? `最佳拟合 ${(state.candidates[0].accuracy * 100).toFixed(1)}%` : "等待程序归纳";
  $("metric-session").textContent = state.session ? state.session.id.toUpperCase() : "未建立";
  $("sieve-observation-count").textContent = observations.length;
  $("sieve-candidate-count").textContent = state.candidates.length || "—";
  const filled = Math.min(8, Math.ceil(observations.length / 2));
  document.querySelectorAll("#coverage-bars i").forEach((bar, index) => bar.classList.toggle("filled", index < filled));
}

function renderCandidates() {
  const body = $("candidate-table");
  if (!state.candidates.length) {
    body.innerHTML = '<tr><td colspan="6" class="empty-row">等待实验数据</td></tr>';
    $("rule-ir-preview").textContent = "等待候选规则…";
    return;
  }
  $("rule-ir-preview").textContent = formatJson(state.candidates[0].expr);
  $("rule-json").value = formatJson(state.candidates[0].expr);
  body.innerHTML = state.candidates.slice(0, 7).map((candidate, index) => `
    <tr>
      <td class="rank">${String(index + 1).padStart(2, "0")}</td>
      <td><code>${escapeHtml(candidate.pretty)}</code></td>
      <td class="${candidate.accuracy >= .99 ? "fit-high" : ""}">${(candidate.accuracy * 100).toFixed(1)}%</td>
      <td>${candidate.complexity}</td>
      <td>${candidate.score.toFixed(2)}</td>
      <td><span class="candidate-status ${index ? "alt" : ""}">${index ? "ALTERNATIVE" : "LEADING"}</span></td>
    </tr>`).join("");
}

function renderObservations() {
  const observations = state.session?.observations || [];
  const body = $("observation-table");
  if (!observations.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty-row">暂无观测</td></tr>';
    return;
  }
  body.innerHTML = observations.map((obs, index) => `
    <tr>
      <td class="rank">${index + 1}</td>
      <td><code>${escapeHtml(obs.episode_id || "default")} / ${obs.step ?? "—"}</code></td>
      <td><code>${escapeHtml(JSON.stringify({ input: obs.input, context: obs.context, state: obs.state }))}</code></td>
      <td class="${obs.output ? "fit-high" : ""}">${obs.output ? "TRUE" : "FALSE"}</td>
      <td>${escapeHtml(obs.source || "unknown")}</td>
    </tr>`).join("");
}

function evidenceRow(icon, title, text, badge) {
  return `<article class="evidence-item"><i>${icon}</i><div><strong>${escapeHtml(title)}</strong><p>${text}</p></div><span>${escapeHtml(badge)}</span></article>`;
}

function renderEvidence(report) {
  const items = [];
  const leading = state.candidates[0];
  if (leading) items.push(evidenceRow("R", "可执行规则已恢复", `<code>${escapeHtml(leading.pretty)}</code>，对已观测样本拟合 ${(leading.accuracy * 100).toFixed(1)}%。`, "RULE IR"));
  if (state.activeQueries.length) {
    const query = state.activeQueries.at(-1);
    items.push(evidenceRow("Δ", "主动反例探针", `候选在 <code>${escapeHtml(JSON.stringify(query.envelope))}</code> 上产生最大预测分歧，已由 Oracle 复验。`, "COUNTEREXAMPLE"));
  }
  if (report?.coverage) {
    const coverage = report.coverage.envelope_coverage;
    items.push(evidenceRow("C", "有限输入域覆盖", `当前声明 domain 覆盖率为 <code>${coverage == null ? "unknown" : `${(coverage * 100).toFixed(1)}%`}</code>；结论不外推到未声明输入。`, "COVERAGE"));
  }
  if (report?.reasons?.length) items.push(evidenceRow("E", "闭环判定依据", escapeHtml(report.reasons[0]), report.confidence?.toUpperCase() || "EVIDENCE"));
  if (report?.limitations?.length) items.push(evidenceRow("!", "外部效度限制", escapeHtml(report.limitations[0]), "LIMITATION"));
  $("evidence-count").textContent = `${items.length} ITEMS`;
  $("evidence-list").innerHTML = items.length ? items.join("") : '<div class="empty-evidence"><svg viewBox="0 0 24 24"><path d="M7 3h10v4h3v14H4V7h3V3Zm2 0v6h6V3M8 14h8M8 17h5"/></svg><strong>尚无实验结论</strong><p>运行研究协议后，这里会记录最小反例、候选分歧与闭环限制。</p></div>';
}

function renderClosure(report) {
  state.closure = report || null;
  if (!report) {
    $("metric-confidence").textContent = "—";
    $("metric-closure").textContent = "尚未形成闭环";
    $("sieve-disagreement").textContent = "—";
    $("sieve-closure").textContent = "—";
    $("verdict-chip").textContent = "WAITING FOR EVIDENCE";
    $("verdict-chip").className = "verdict-chip";
    renderEvidence(null);
    return;
  }
  const confidenceMap = { high: "高", medium: "中", low: "低" };
  $("metric-confidence").textContent = `${(report.closure_score * 100).toFixed(1)}%`;
  $("metric-closure").textContent = `${closureLabels[report.closure_status] || report.closure_status} · ${confidenceMap[report.confidence] || report.confidence}置信`;
  $("sieve-disagreement").textContent = report.hypotheses?.max_disagreement == null ? "—" : Number(report.hypotheses.max_disagreement).toFixed(3);
  $("sieve-closure").textContent = `${(report.closure_score * 100).toFixed(1)}%`;
  $("verdict-chip").textContent = closureLabels[report.closure_status]?.toUpperCase() || report.closure_status;
  $("verdict-chip").className = `verdict-chip ${["identified", "observationally_closed"].includes(report.closure_status) ? "closed" : "risk"}`;
  $("closure-score").textContent = `${(report.closure_score * 100).toFixed(1)}%`;
  $("closure-classes").textContent = report.hypotheses?.behavior_class_count ?? "—";
  $("closure-disagreement").textContent = report.hypotheses?.max_disagreement ?? "—";
  $("closure-coverage").textContent = report.coverage?.envelope_coverage ?? "—";
  $("closure-json").textContent = formatJson(report);
  renderEvidence(report);
}

function renderAll() {
  renderMetrics();
  renderCandidates();
  renderObservations();
  renderClosure(state.closure);
  const enabled = Boolean(state.session) && !state.running;
  ["probe", "manual-true", "manual-false", "run-search", "run-closure"].forEach((id) => $(id).disabled = !enabled);
  $("suggest-query").disabled = !enabled || state.candidates.length < 2;
}

async function createSession() {
  const scenario = selectedScenario();
  state.session = await api("/api/sessions", { method: "POST", body: JSON.stringify({ scenario_id: scenario.id }) });
  state.candidates = [];
  state.closure = null;
  state.activeQueries = [];
  renderFields(state.session.scenario.fields);
  renderAll();
}

async function refreshSession() {
  if (!state.session) return;
  state.session = await api(`/api/sessions/${state.session.id}`);
  state.candidates = state.session.candidates || [];
  state.closure = state.session.last_closure_report || state.closure;
  renderAll();
}

function cartesianCases(fields, limit = 8) {
  let cases = [{ input: {}, context: {}, state: {} }];
  for (const field of fields) {
    const next = [];
    for (const row of cases) for (const value of field.domain || [""]) {
      const clone = structuredClone(row);
      setPath(clone, field.path, value);
      next.push(clone);
    }
    cases = next;
  }
  if (cases.length <= limit) return cases;
  const indexes = new Set([0, cases.length - 1]);
  for (let i = 1; indexes.size < limit && i < limit * 3; i++) indexes.add(Math.round(i * (cases.length - 1) / (limit - 1)));
  return [...indexes].sort((a, b) => a - b).slice(0, limit).map((index) => cases[index]);
}

async function probeEnvelope(envelope) {
  return api(`/api/sessions/${state.session.id}/probe`, { method: "POST", body: JSON.stringify(envelope) });
}

async function seedProtocol(scenario) {
  if (scenario.stateful && scenario.fields.length === 1 && scenario.fields[0].path === "input.action") {
    const sequences = [["verify", "commit"], ["wait", "commit"], ["verify", "cancel"], ["commit", "commit"]];
    for (let episode = 0; episode < sequences.length; episode++) {
      for (let step = 0; step < sequences[episode].length; step++) {
        await probeEnvelope({ input: { action: sequences[episode][step] }, context: {}, state: {}, episode_id: `seed-${episode + 1}`, step });
      }
      await sleep(35);
    }
    return;
  }
  const cases = cartesianCases(scenario.fields, 8);
  for (let index = 0; index < cases.length; index++) {
    await probeEnvelope({ ...cases[index], episode_id: `seed-${index + 1}`, step: 0 });
    if (index % 2 === 1) await sleep(35);
  }
}

function searchPayload() {
  return {
    max_depth: Number($("max-depth").value),
    beam_width: Number($("beam-width").value),
    history_depth: Number($("history-depth").value),
  };
}

async function runSearch() {
  const result = await api(`/api/sessions/${state.session.id}/search`, { method: "POST", body: JSON.stringify(searchPayload()) });
  state.candidates = result.candidates || [];
  renderAll();
  return result;
}

async function runClosure() {
  const payload = {
    max_cases: Number($("closure-max-cases").value),
    coverage_threshold: Number($("coverage-threshold").value),
    goal_mode: $("goal-mode").value,
    history_depth: Number($("history-depth").value),
    max_depth: Number($("max-depth").value),
    beam_width: Number($("beam-width").value),
    auto_search: true,
  };
  const report = await api(`/api/sessions/${state.session.id}/closure/analyze`, { method: "POST", body: JSON.stringify(payload) });
  state.closure = report;
  renderClosure(report);
  return report;
}

async function runResearchProtocol() {
  if (state.running) return;
  state.running = true;
  $("run-protocol").disabled = true;
  setRunStatus("running", "RUNNING");
  setProtocolStep(1, "正在构造边界值、对抗字符串与状态序列探针…");
  try {
    await createSession();
    await seedProtocol(selectedScenario());
    await refreshSession();

    setProtocolStep(2, "正在将观测行为压缩为可执行 Rule IR 候选…");
    await runSearch();
    await sleep(180);

    setProtocolStep(3, "正在用最大预测熵选择后续实验，筛除行为不一致假设…");
    for (let round = 0; round < 3 && state.candidates.length >= 2; round++) {
      const result = await api(`/api/sessions/${state.session.id}/suggest`);
      if (!result.suggestion) break;
      const suggestion = result.suggestion;
      const envelope = { input: suggestion.input, context: suggestion.context, state: suggestion.state, episode_id: `active-${round + 1}`, step: 0 };
      state.activeQueries.push({ envelope: { input: suggestion.input, context: suggestion.context, state: suggestion.state }, entropy: suggestion.disagreement });
      await probeEnvelope(envelope);
      await runSearch();
      await sleep(120);
    }

    setProtocolStep(4, "正在计算行为等价类、域覆盖与闭环证据条件…");
    await runClosure();
    await refreshSession();
    state.running = false;
    setProtocolStep(4, "协议完成：已生成可复现规则、主动反例与有限域闭环报告。");
    setRunStatus("complete", "COMPLETE");
    $("metric-risk").textContent = selectedScenario().severity || "已评估";
    renderAll();
    toast("研究协议运行完成，证据账本已更新。", "success");
  } catch (error) {
    state.running = false;
    setRunStatus("failed", "FAILED");
    $("run-message").textContent = `协议中止：${error.message}`;
    toast(error.message, "error");
  } finally {
    $("run-protocol").disabled = false;
  }
}

function showView(name) {
  const labels = {
    overview: ["OVERVIEW", "黑盒规则归纳与漏洞筛查", "用可控的 JavaScript 缺陷程序研究大模型的规则抽象、主动实验与反例发现能力。"],
    training: ["MODEL TRAINING", "专用规则模型训练方案", "以 GPT-2 风格共享主干、稀疏 MoE 专家和主动强化学习专攻黑盒规则难题。"],
    corpus: ["DEFECT CORPUS", "JavaScript 缺陷语料库", "以可执行、可变异、按规则族隔离的程序构建科研训练语料。"],
    runs: ["EXPERIMENT RUNS", "实验运行记录", "追踪每次查询、候选收敛、反例发现与闭环证据。"],
    hypotheses: ["RULE HYPOTHESES", "可执行规则假设", "比较 Rule IR 候选的行为拟合、复杂度与等价类。"],
    evidence: ["VULNERABILITY EVIDENCE", "漏洞证据仓", "只沉淀能够由黑盒 Oracle 重放验证的安全结论。"],
  };
  const [breadcrumb, title, subtitle] = labels[name] || labels.overview;
  $("view-breadcrumb").textContent = breadcrumb;
  $("view-title").textContent = title;
  $("view-subtitle").textContent = subtitle;
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
}

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
document.querySelectorAll(".back-overview").forEach((button) => button.addEventListener("click", () => showView("overview")));
document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === button));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${button.dataset.tab}`));
}));

$("scenario-select").addEventListener("change", () => {
  renderScenario();
  state.session = null;
  state.candidates = [];
  state.closure = null;
  state.activeQueries = [];
  setRunStatus("idle", "IDLE");
  setProtocolStep(0, "语料已切换。运行协议以建立新的独立实验会话。");
  renderAll();
});
$("run-protocol").addEventListener("click", runResearchProtocol);

$("probe").addEventListener("click", async () => {
  try {
    const result = await probeEnvelope(envelopeFromForm());
    $("oracle-result").className = `console-message ${result.output ? "good" : "bad"}`;
    $("oracle-result").textContent = `ORACLE → ${result.output ? "TRUE" : "FALSE"}`;
    await refreshSession();
  } catch (error) { toast(error.message, "error"); }
});

async function manualObservation(output) {
  try {
    await api(`/api/sessions/${state.session.id}/observe`, { method: "POST", body: JSON.stringify({ ...envelopeFromForm(), output, source: "manual" }) });
    await refreshSession();
  } catch (error) { toast(error.message, "error"); }
}
$("manual-true").addEventListener("click", () => manualObservation(true));
$("manual-false").addEventListener("click", () => manualObservation(false));
$("run-search").addEventListener("click", () => runSearch().catch((error) => toast(error.message, "error")));
$("run-closure").addEventListener("click", () => runClosure().catch((error) => toast(error.message, "error")));
$("suggest-query").addEventListener("click", async () => {
  try {
    const result = await api(`/api/sessions/${state.session.id}/suggest`);
    if (!result.suggestion) return void ($("suggestion").textContent = result.reason);
    fillForm(result.suggestion);
    $("suggestion").textContent = `ACTIVE QUERY → ${JSON.stringify({ input: result.suggestion.input, context: result.suggestion.context, state: result.suggestion.state })}`;
  } catch (error) { toast(error.message, "error"); }
});

$("export-report").addEventListener("click", () => {
  if (!state.session) return toast("当前还没有可导出的实验运行。", "error");
  const payload = { exported_at: new Date().toISOString(), scenario: selectedScenario(), session: state.session, closure: state.closure, active_queries: state.activeQueries };
  const blob = new Blob([formatJson(payload)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `sift-${selectedScenario().id}-${state.session.id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
  toast("研究报告已导出。", "success");
});

(async () => {
  try {
    const [health, scenarios] = await Promise.all([api("/api/health"), api("/api/scenarios")]);
    state.scenarios = scenarios;
    $("scenario-select").innerHTML = scenarios.map((scenario) => `<option value="${escapeHtml(scenario.id)}">${escapeHtml(scenario.name)}</option>`).join("");
    if (scenarios.some((scenario) => scenario.id === "js_truthy_access")) $("scenario-select").value = "js_truthy_access";
    $("api-status").textContent = `NODE ONLINE · v${health.version}`;
    document.querySelector(".sidebar-note").classList.add("online");
    $("run-protocol").disabled = false;
    renderScenario();
    renderAll();
  } catch (error) {
    $("api-status").textContent = "NODE OFFLINE";
    toast(`无法连接研究节点：${error.message}`, "error");
  }
})();
