# PG-255 人员交付与数据采集单

## 一句话结论

AI 已经接入真实的本地 GET/POST 发送路径，可以在新鲜 Pikachu 容器里决定是否发出受控 syntax probe。PG-255 共 14 个全新容器、10 GET/4 POST，AI 发出 14 次，独立 reference 发出 14 次；其中 8 次通过类型化 SQL response-shape oracle，6 次保持未确认。发送探针不等于漏洞成立，当前没有任何任意网址或公网能力结论。

## 人应该如何读一条记录

| 状态 | 含义 | 人的下一步 |
|---|---|---|
| `SEND_PROBE` | AI 的 pre-probe gate 允许发一个只读、已绑定字段的探针 | 检查请求方法、字段、fresh reset 和 scope；不要直接称为漏洞 |
| `EFFECT_OBSERVED` | 响应投影出现了候选与 reference 一致的可观察差异 | 查看 matched negative 是否干净，确认差异不是环境噪声 |
| `CONFIRMED_POSITIVE` | fresh reset、reference、negative、源码哈希、证据哈希和 typed oracle 全部通过 | 只在授权本地靶场登记为“本地效果确认”，再决定是否制作脱敏训练样本 |
| `ABSTAIN_ORACLE_GAP` | 发送被拒绝，或效果缺少可判定 oracle | 补 evaluator/复放信息；不要补猜测标签，不要提升长期记忆 |

本轮的 `CONFIRMED_POSITIVE` 仅覆盖：`POST /vul/sqli/sqli_id.php`、`GET /vul/sqli/sqli_search.php`、`GET /vul/sqli/sqli_str.php`、`GET /vul/sqli/sqli_x.php`。blind boolean、timing 和 widebyte 没有足够证据时，必须保持未确认。

## 给采集员的任务

1. 只使用项目登记的 loopback Docker 靶场和 pinned image；每个路由新建无 volume/bind mount 的容器。
2. 先记录路由、方法、字段、状态类别、重定向跳数和 baseline/control 投影；GET 与 POST 分开统计。
3. 按固定顺序执行 `baseline → control → AI candidate → independent reference → matched negative`。AI 的决定必须发生在 candidate 发送之前。
4. 只保存请求结构、编码类别、状态/长度桶、响应投影、marker/payload/response 哈希和证据哈希；原始请求值、原始响应正文只在本地请求期间存在，不写入数据集。
5. 如果数据库健康门、源码哈希、负对照或 evaluator 缺失，把记录标记为 `ABSTAIN_ORACLE_GAP` 或 `quarantine`，不要自行猜正负。

## 给复核员的任务

- 检查 AI candidate 与独立 reference 是否同属预注册 probe class。
- 检查 negative 是否没有同样的效果信号；如果 negative 也变化，判定为环境/基线问题。
- 检查 fresh reset、容器无卷、数据库健康门、源码 hash、evidence hash 是否齐全。
- 对 `CONFIRMED_POSITIVE` 只写“本地 typed effect confirmed”；禁止写成“任意网站存在漏洞”或“payload 对所有 WAF 有效”。
- 对 blind/timing/escape 等没有 evaluator 的族，明确写出缺失原因，而不是用错误页或时间猜测替代 oracle。

## 给训练员的任务

训练样本必须先经过最终判官：

- `gold`：完整来源、GET/POST 字段、独立复放、typed effect 或明确 abstain、负对照和跨 seed/route 留出均通过。
- `hard_negative`：失败阶段和 model/environment 归因完整，并有可复现修复或 abstain 轨迹。
- `silver`：只有完整的安全探针上下文；不得训练“漏洞阳性”或 payload 记忆。
- `quarantine`：缺字段、缺来源、不可复现、oracle 不足；修复后以新 parent_record_id 追加，不能覆盖原记录。

训练重点不是收集更多 URL，而是收集可比较的过程三元组：

`失败/观测 → 最小下一步探针 → 独立结果/修复`。

优先扩充以下维度：

- 同一路由的 GET/POST 成对样本；同一字段的不同编码/表面，但抽象 Rule IR 相同。
- 正例、匹配阴性、oracle 不可用三者成套出现。
- 不同 seed、不同实现、不同路由族的完整留出；不能随机切相同模板。
- 明确失败信息：数据库不可用、字段绑定失败、重定向差异、响应形状差异、reference 不一致、模型误发/漏发。
- 每次修复只改变一个最小 token/Rule-IR 槽位，并在 fresh target 上重放。

不要采集：公网未授权目标、真实凭据、破坏性写入、时间延迟探针、外联回调、只保存“看起来像成功”的截图、没有负对照的单条成功请求。

## 最小交付包

- 报告：`pg255_pikachu_fixed_sql_pg254_replay_report_v1.json`
- 协议：`pg255_pikachu_fixed_sql_pg254_replay_protocol_v1.json`
- 轨迹：`pg255_pikachu_fixed_sql_pg254_replay_trace_v1.json`
- 运行器：`scripts/run_pg255_pikachu_fixed_sql_pg254_replay.py`
- 最终规则：`improvement_rules.json` 中的 `pg255_pikachu_fixed_sql_pg254_replay`

交付时优先展示每条记录的状态卡和证据哈希；需要查看真实 wire 时重新运行本地脚本，让它只在终端临时显示，不从报告复制原始 payload。
