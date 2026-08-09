# SIFT 前端漏洞靶场审计

日期：2026-08-01  
目标：本地 `http://127.0.0.1:3000/playground`  
执行面：浏览器侧 JavaScript，仅本地受控载荷

## 结果

5 个探针全部复现了“实现行为与安全策略不一致”。测试没有访问外部目标，没有发生真实跳转，也没有执行注入脚本；DOM 注入只在脱离文档的容器中验证是否创建了非预期节点。

| Finding | CWE | 最小反例 | 错误行为 | 抽象规则 |
|---|---|---|---|---|
| 数值被误当授权位 | CWE-863 | `{"role":"guest","quota":-1}` | `true`，期望 `false` | `authorization_decision := role_is_admin OR to_boolean(quota)` |
| 子串替代 Origin 校验 | CWE-601 | `https://trusted.com.evil.test/phish` | `true`，期望 `false` | `redirect_allowed := substring(url, trusted_host)` |
| 优惠边界漂移 | CWE-193 | `{"member":true,"total":100}` | `false`，期望 `true` | `coupon_allowed := member AND total GT threshold` |
| 消息源后缀混淆 | CWE-346 | `https://eviltrusted.com` | `true`，期望 `false` | `message_trusted := suffix(origin, trusted_host)` |
| 文本被解释为 HTML | CWE-79 | `</p><span data-sift-injected>probe</span><p>` | 创建新 DOM 节点 | `dom_nodes := html_parse(untrusted_text)` |

## 暴露出的研究问题

前三类已有对应语料族：`truthiness_gate`、`substring_origin`、`numeric_boundary`。后两类暴露了当前研究覆盖缺口：

1. 训练课程没有独立的 `postmessage_origin` 与 `dom_sink_injection` 规则族；
2. Rule IR 只有字符串 `contains / starts_with / ends_with`，没有结构化 URL 的 `scheme / hostname / origin` 操作，因此模型容易把字符串相关性当作安全边界；
3. Rule IR 没有 `source → transform → sink`、编码上下文和 sanitizer 语义，无法表达 DOM XSS 的因果链；
4. 当前黑盒 Oracle 主要返回布尔值，DOM 结构变化、导航意图和消息信任决策尚未成为一等观测；
5. 现有闭环分数不能代表浏览器语义已经闭环，需要加入浏览器版本、DOM 快照和 URL 解析结果作为证据指针。

## 根因判断

这是实验表示与数据覆盖问题，不是算力或前端工程吞吐问题。下一步应先扩充 Rule IR 与两个小型合成规则族，并做族外消融；在它们通过前，不应扩大模型或数据规模。

## 建议的最小下一实验

- 新增结构化 `parse_url` 观测与 `origin_eq` 规则节点；
- 新增 `dom_sink_injection` 的 source/sink/context 三元组，但继续禁止真实脚本执行；
- 分别留出完整 `postmessage_origin`、`dom_sink_injection` 家族；
- 对比字符串 Rule IR、结构化 URL/DOM Rule IR 两组，预注册指标为 Counterexample@10 与族外行为一致率；
- 只有结构化表示显著改善且无误报退化，才进入正式训练课程。
