"use client";

import { useMemo, useState } from "react";
import styles from "./pg389-js-chain-lab.module.css";

type Action = "REPAIR" | "ASK" | "ABSTAIN" | "SELECT";

type ChainCase = {
  id: string;
  title: string;
  family: string;
  transport: "GET" | "POST";
  source: string;
  chain: string[];
  filter: string;
  guard: string;
  sink: string;
  observations: string[];
  action: Action;
  repair: string;
  oracle: string;
  boundary: string;
};

const cases: ChainCase[] = [
  {
    id: "query-decode-filter",
    title: "解码后过滤",
    family: "query / text sink",
    transport: "GET",
    source: "location_search",
    chain: ["query_parse", "field_extract", "percent_decode"],
    filter: "after_decode",
    guard: "filter_before_sink",
    sink: "text_sink",
    observations: ["input_seen", "decoded_once", "filter_rejected", "sink_not_reached"],
    action: "REPAIR",
    repair: "encoding",
    oracle: "bounded_text_shape",
    boundary: "只记录阶段和形状；不保存具体输入。",
  },
  {
    id: "filter-before-decode",
    title: "解码前过滤",
    family: "query / filter order",
    transport: "GET",
    source: "location_search",
    chain: ["query_parse", "field_extract", "percent_decode"],
    filter: "before_decode",
    guard: "filter_before_sink",
    sink: "text_sink",
    observations: ["input_seen", "filter_rejected", "decoder_not_reached", "sink_not_reached"],
    action: "SELECT",
    repair: "encoding",
    oracle: "bounded_text_shape",
    boundary: "只允许一变量 fixture 变体，不生成任意字符串。",
  },
  {
    id: "double-decode",
    title: "双层解码顺序",
    family: "query / order-sensitive",
    transport: "GET",
    source: "location_search",
    chain: ["query_parse", "field_extract", "percent_decode", "percent_decode"],
    filter: "between_decode_steps",
    guard: "guard_after_normalize",
    sink: "text_sink",
    observations: ["input_seen", "decoded_once", "guard_checked", "decoded_twice", "sink_shape_observed"],
    action: "REPAIR",
    repair: "encoding",
    oracle: "bounded_marker_shape",
    boundary: "第二次解码只以抽象阶段表示。",
  },
  {
    id: "json-parser",
    title: "JSON 解析边界",
    family: "JSON / structured sink",
    transport: "POST",
    source: "form_input",
    chain: ["json_parse", "field_extract"],
    filter: "after_parse",
    guard: "schema_before_sink",
    sink: "structured_value_sink",
    observations: ["input_seen", "parsed", "schema_rejected", "sink_not_reached"],
    action: "ASK",
    repair: "syntax",
    oracle: "parser_error_shape",
    boundary: "解析失败只回传有界错误类别。",
  },
  {
    id: "form-route",
    title: "表单解码与路由",
    family: "form / route guard",
    transport: "POST",
    source: "form_input",
    chain: ["form_decode", "trim"],
    filter: "before_route",
    guard: "allowlist_before_route",
    sink: "route_state_sink",
    observations: ["input_seen", "decoded_once", "guard_rejected", "route_not_taken"],
    action: "ASK",
    repair: "encoding",
    oracle: "bounded_redirect_shape",
    boundary: "不访问外部地址；只观察本地路由形状。",
  },
  {
    id: "fragment-guard",
    title: "Fragment 一次解码",
    family: "hash / attribute sink",
    transport: "GET",
    source: "location_hash",
    chain: ["fragment_parse", "percent_decode"],
    filter: "after_decode",
    guard: "sink_guard_before_render",
    sink: "attribute_sink",
    observations: ["input_seen", "decoded_once", "guard_rejected", "sink_not_reached"],
    action: "ASK",
    repair: "none",
    oracle: "bounded_attribute_shape",
    boundary: "属性 sink 缺上下文时保持 ASK。",
  },
  {
    id: "normalize-allowlist",
    title: "规范化后白名单",
    family: "form / normalization",
    transport: "POST",
    source: "form_input",
    chain: ["form_decode", "trim", "casefold"],
    filter: "after_normalize",
    guard: "allowlist_before_sink",
    sink: "text_sink",
    observations: ["input_seen", "trimmed", "casefolded", "guard_rejected", "sink_not_reached"],
    action: "SELECT",
    repair: "normalization",
    oracle: "bounded_text_shape",
    boundary: "只保留规范化阶段，不保存词面。",
  },
  {
    id: "escape-text",
    title: "转义后的文本 sink",
    family: "query / safe rendering",
    transport: "GET",
    source: "location_search",
    chain: ["query_parse", "field_extract", "percent_decode", "html_escape"],
    filter: "escape_at_sink",
    guard: "escape_before_sink",
    sink: "text_sink",
    observations: ["input_seen", "decoded_once", "escaped", "text_shape_observed"],
    action: "ABSTAIN",
    repair: "none",
    oracle: "bounded_text_shape",
    boundary: "文本 sink 不被当作可执行 sink。",
  },
  {
    id: "scheme-guard",
    title: "协议白名单",
    family: "form / URL attribute",
    transport: "POST",
    source: "form_input",
    chain: ["form_decode", "trim", "scheme_parse"],
    filter: "before_attribute_sink",
    guard: "scheme_allowlist_before_sink",
    sink: "url_attribute_sink",
    observations: ["input_seen", "decoded_once", "scheme_checked", "guard_rejected", "sink_not_reached"],
    action: "ABSTAIN",
    repair: "none",
    oracle: "bounded_redirect_shape",
    boundary: "外部 scheme 被硬阻断。",
  },
  {
    id: "parser-short-circuit",
    title: "解析错误短路",
    family: "JSON / parser boundary",
    transport: "POST",
    source: "form_input",
    chain: ["json_parse"],
    filter: "parser_boundary",
    guard: "parser_before_filter",
    sink: "structured_value_sink",
    observations: ["input_seen", "parser_error", "filter_not_reached", "sink_not_reached"],
    action: "REPAIR",
    repair: "syntax",
    oracle: "parser_error_shape",
    boundary: "只允许有界 parser_error 类别。",
  },
  {
    id: "persistent-block",
    title: "持久化状态硬门",
    family: "state / write guard",
    transport: "POST",
    source: "form_input",
    chain: ["form_decode", "trim"],
    filter: "before_state_write",
    guard: "persistence_block_before_sink",
    sink: "persistent_state_sink",
    observations: ["input_seen", "decoded_once", "persistence_guard", "write_not_attempted"],
    action: "ABSTAIN",
    repair: "none",
    oracle: "no_write_shape",
    boundary: "不执行持久化写入，缺授权直接 ABSTAIN。",
  },
  {
    id: "dynamic-code-block",
    title: "动态代码硬门",
    family: "code / blocked sink",
    transport: "POST",
    source: "form_input",
    chain: ["form_decode", "dynamic_code_boundary"],
    filter: "before_code_sink",
    guard: "dynamic_code_block_before_sink",
    sink: "dynamic_code_sink",
    observations: ["input_seen", "decoded_once", "code_guard_rejected", "sink_not_reached"],
    action: "ABSTAIN",
    repair: "none",
    oracle: "no_execution_shape",
    boundary: "动态代码、外连和持久化均不进入发送许可。",
  },
];

const actionCopy: Record<Action, string> = {
  REPAIR: "修复一处链路变量，再观察反馈",
  ASK: "缺信息，先请求上下文",
  ABSTAIN: "边界不满足，安全拒绝",
  SELECT: "选择一个 fixture-bound 形状",
};

export default function Pg389JsChainLab() {
  const [selected, setSelected] = useState(0);
  const [showProjection, setShowProjection] = useState(true);
  const item = cases[selected];
  const stats = useMemo(() => ({
    get: cases.filter((entry) => entry.transport === "GET").length,
    post: cases.filter((entry) => entry.transport === "POST").length,
    guarded: cases.filter((entry) => entry.guard !== "none_observed").length,
  }), []);

  return (
    <main className={styles.page}>
      <header className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>PG-389 · JS CHAIN LAB</p>
          <h1>解码顺序、过滤阶段与 sink 语义</h1>
          <p className={styles.subtitle}>
            把“黑盒反馈”拆成可观察的阶段序列，让模型先判断缺什么、哪里短路、下一步是否应该 ASK/REPAIR。
          </p>
        </div>
        <div className={styles.statusPill}><span /> candidate-only · local abstract</div>
      </header>

      <section className={styles.metrics} aria-label="实验概览">
        <div><strong>{cases.length}</strong><span>链路案例</span></div>
        <div><strong>{stats.get}/{stats.post}</strong><span>GET / POST</span></div>
        <div><strong>{stats.guarded}</strong><span>guard 变体</span></div>
        <div><strong>0</strong><span>原始 wire 进入模型</span></div>
      </section>

      <section className={styles.workspace}>
        <aside className={styles.caseRail}>
          <div className={styles.railHeader}><span>CASE MATRIX</span><small>12 patterns</small></div>
          {cases.map((entry, index) => (
            <button key={entry.id} className={`${styles.caseButton} ${index === selected ? styles.active : ""}`} onClick={() => setSelected(index)}>
              <span className={styles.caseIndex}>{String(index + 1).padStart(2, "0")}</span>
              <span><b>{entry.title}</b><small>{entry.family}</small></span>
              <em>{entry.transport}</em>
            </button>
          ))}
        </aside>

        <div className={styles.detail}>
          <div className={styles.detailTop}>
            <div><p className={styles.kicker}>{item.family} · {item.source}</p><h2>{item.title}</h2></div>
            <span className={styles.method}>{item.transport}</span>
          </div>

          <div className={styles.chainCard}>
            <div className={styles.cardTitle}><span>ABSTRACT EXECUTION CHAIN</span><small>ordered / no source text</small></div>
            <div className={styles.chainFlow}>
              {item.chain.map((step, index) => <div className={styles.chainStep} key={`${step}-${index}`}><span>{index + 1}</span><b>{step}</b>{index < item.chain.length - 1 && <i>→</i>}</div>)}
            </div>
            <div className={styles.chainMeta}>
              <div><small>FILTER STAGE</small><strong>{item.filter}</strong></div>
              <div><small>GUARD PRECEDENCE</small><strong>{item.guard}</strong></div>
              <div><small>SINK CONTEXT</small><strong>{item.sink}</strong></div>
            </div>
          </div>

          <div className={styles.columns}>
            <div className={styles.observationCard}>
              <div className={styles.cardTitle}><span>BLACK-BOX OBSERVATIONS</span><small>typed shape only</small></div>
              <ol className={styles.observationList}>{item.observations.map((observation, index) => <li key={observation}><span>{index + 1}</span><b>{observation}</b></li>)}</ol>
            </div>
            <div className={styles.decisionCard}>
              <div className={styles.cardTitle}><span>RULE-IR DECISION</span><small>model target</small></div>
              <div className={`${styles.action} ${styles[`action${item.action}`]}`}><strong>{item.action}</strong><span>{actionCopy[item.action]}</span></div>
              <div className={styles.decisionRows}><div><small>repair_action</small><b>{item.repair}</b></div><div><small>oracle_shape</small><b>{item.oracle}</b></div></div>
            </div>
          </div>

          <div className={styles.boundary}>{item.boundary}</div>
          <button className={styles.toggle} onClick={() => setShowProjection((value) => !value)}>{showProjection ? "隐藏" : "显示"} projection tokens</button>
          {showProjection && <div className={styles.tokens}><span>[BOS]</span><span>source={item.source}</span>{item.chain.map((step, index) => <span key={step + index}>decoder_step_{index + 1}={step}</span>)}<span>filter_stage={item.filter}</span><span>guard_precedence={item.guard}</span><span>sink_context={item.sink}</span><span>[CTX_END]</span></div>}
        </div>
      </section>

      <footer className={styles.footer}><span>PG-389 abstract projection</span><span>原始脚本 / 输入 / 响应体 / evaluator 答案均留在本地审阅侧</span><span>training=false · promotion=false</span></footer>
    </main>
  );
}
