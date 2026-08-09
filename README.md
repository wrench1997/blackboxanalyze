# SIFT AI Research Lab 0.3

面向 **JavaScript 黑盒规则归纳与漏洞发现** 的科研原型。研究对象不是普通“闯关游戏”，而是把带缺陷的 JS 中间程序抽象为有限、可执行、可证伪的游戏规则，再评估 AI 是否能用尽可能少的查询恢复规则并发现安全反例。

## 研究仪表盘

Web 工作台围绕一条可复现实验链组织：

```text
JS 缺陷语料 → 语义/游戏规则抽象 → 黑盒行为探针
→ Rule IR 候选归纳 → 最大分歧主动查询 → 漏洞证据与闭环报告
```

内置科研语料包括：

- JavaScript truthiness 引发的授权绕过；
- 字符串包含校验引发的开放重定向；
- `>` / `>=` 边界漂移；
- 未绑定挑战或会话的序列重放窗口。

点击“运行研究协议”会创建独立 Run，生成边界/对抗探针，执行 Beam + MDL 规则归纳，再用候选最大分歧选择追加实验。仪表盘明确展示实验预算、候选假设、最小反例、有限域覆盖、行为等价类与结论限制。

新版前端位于 `frontend/`，使用 Next.js App Router。FastAPI 保留为研究引擎：

```bash
# 终端 1：研究 API
uvicorn app.main:app --host 127.0.0.1 --port 8080

# 终端 2：Next.js 前端
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:3000`。Next.js 会把 `/api/*` 代理到本地 FastAPI。

## 语言无关规则层

源码、AST、字节码和黑盒轨迹最终统一成 Common Semantic Rule。长期记忆保存的是 coercion、boundary、state、history、invariant、effect 和 evidence 等抽象语义，而不是绑定某种语言的表面 token。整数模型、溢出、异常、求值顺序和 nullish 行为作为显式语言参数保留。

规则发现支持三种有来源标记的模式：明文抽取、静态与运行时联合的灰盒消解、纯输入输出驱动的黑盒归纳。对应规范见 `research/common_semantic_rule.schema.json`、`research/discovery_modes.json` 与 `research/RESEARCH_RULES.md`。

一个可运行的“黑盒规则摸索 + 闭环判断”Demo。它把任意语言程序的可观察行为统一成：

```text
字段域 + Episode 观测轨迹 + Rule IR 候选程序
+ 主动实验 + 行为等价类 + 状态转移图 + 闭环报告
```

它不会宣称可以数学上证明任意程序完全等价，但会明确区分：

- 还有候选分歧，应该继续测试；
- 当前有限输入域内已经观测闭环；
- 只有一个候选规则且覆盖足够；
- 同一上下文输出冲突，疑似隐藏状态、随机性或脏数据；
- 当前 DSL 或搜索预算无法解释数据；
- 状态图进入无目标的终端强连通分量，疑似或确认死路。

## 已实现

### 程序归纳

- JSON Rule IR：字段、历史字段、比较、逻辑、算术、取模、字符串、长度、正则、条件表达式。
- 多命名空间：`input.*`、`context.*`、`state.*`。
- Episode-aware 历史：不同回合互不污染，支持 `prev[offset].path`。
- 自动原子条件生成：枚举、布尔、阈值、取模、字符串片段、数组长度与包含。
- Beam Search 组合搜索。
- 保留多个“当前样本上等价、未见输入上可能不同”的结构候选，避免假闭环。
- MDL 风格复杂度惩罚，优先简单规则。
- 最大分歧主动查询。
- Oracle、自定义 Rule IR、外部黑盒手工导入。
- 隐藏测试与反例展示。

### Closure Analyzer

- 可见上下文冲突检测与输出熵。
- 候选规则完整拟合检查。
- 在声明的有限 domain 上生成行为向量并划分行为等价类。
- 计算最大候选分歧与下一条高价值实验。
- 输入域覆盖率、字段取值覆盖率、数值边界覆盖率。
- 字段必要度启发式分析。
- `state → state_after` 状态图构造。
- Tarjan 强连通分量分析。
- 无目标终端 SCC 检测。
- `available_actions` 动作覆盖证明。
- `terminal=true` 显式终态支持。
- 多轮闭环指纹稳定度。
- 综合闭环分数、置信度、证据条件与建议。

## 闭环状态

| 状态 | 含义 |
|---|---|
| `insufficient_data` | 数据不足。 |
| `open` | 仍存在能区分候选规则的新实验。 |
| `identified` | 只剩一个一致候选，且覆盖达到阈值。 |
| `observationally_closed` | 语法规则可能不唯一，但在声明输入域内行为等价。 |
| `observationally_closed_low_coverage` | 暂时无分歧，但覆盖不足，不能证明真实黑盒闭环。 |
| `deadlocked` | 无目标终端 SCC，且动作覆盖完整或显式终态。 |
| `suspected_deadlock` | 看起来无出口，但尚有动作未验证。 |
| `context_incomplete_or_nondeterministic` | 同一完整可见条件出现不同输出。 |
| `dsl_or_search_insufficient` | 当前规则语言或搜索预算无法完整解释数据。 |
| `budget_or_domain_limited` | 没找到分歧，但候选仍不唯一，可能是 domain 或枚举预算过小。 |

## 直接运行

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

打开：`http://localhost:8080`

API 文档：`http://localhost:8080/docs`

### PG-03 本地回放采集

只读采集器固定使用 `http://127.0.0.1:3100` 的本地 replay adapter。每条样本先记录 fresh reset，再保存状态码、响应哈希、bounded JSON shape、oracle projection 和 Rule IR；不会保存原始 body，也不会访问公网。

```bash
python scripts/run_pg03_local_replay.py
```

结果写入 `research/payload_replay_catalog_v1.json`、`research/payload_replay_collector_v1.json` 和 `research/payload_replay_collector_v1.md`。来源隔离、族外 holdout 和 fail-closed abstention 指标都记录在报告中。

### PG-03 Rule IR 解码实验

使用上述 Catalog 训练一个 34,661 参数的 CUDA/CPU 小型 MLP。模型只接收脱敏的 probe/响应形状，输出经过 grammar 校验的家族级 Rule IR；confidence、top-2 margin 和新 surface 距离三道门任一失败就 abstain。

```bash
python scripts/train_pg03_rule_ir_decoder.py
```

训练与族外结果见 `research/pg03_rule_ir_decoder_v1.json` / `research/pg03_rule_ir_decoder_v1.md`，协议见 `research/pg03_rule_ir_decoder_protocol_v1.json`，checkpoint 位于 `artifacts/pg03-rule-ir-decoder/`。`exit_found` 仅表示候选家族与本地迷宫语义一致，不等于真实漏洞确认；族外结构同形导致的误报会保留在报告中，不能被平均准确率掩盖。

V2 强化基线：`CatalogRuleIRDecoderV2` 使用 surface/context 双塔、噪声视图增强、监督式对比损失和最小族支持门。它把结构不可辨识的族外样本强制置为 abstain；结果见 `research/pg03_rule_ir_decoder_v2.md`，训练命令为：

```bash
python scripts/train_pg03_rule_ir_decoder_v2.py
```

## Docker

```bash
docker compose up --build
```

修改宿主机端口：

```bash
RULE_LAB_PORT=18080 docker compose up --build
```

## 推荐使用流程

1. 创建 Oracle 或手工会话。
2. 录入正例、反例和边界输入。
3. 有状态程序为每条记录填写 `episode_id`、`step`、`state`、`state_after`。
4. 搜索候选规则。
5. 运行 Closure Analyzer。
6. 状态为 `open` 时，执行报告里的 `best_disagreement_case`。
7. 状态为 `suspected_deadlock` 时，补测终端 SCC 中未覆盖的动作。
8. 新数据录入后重新搜索、重新闭环分析。

## 观测格式

最小格式：

```json
{
  "input": {"x": 2},
  "context": {},
  "state": {},
  "output": true
}
```

有状态推荐格式：

```json
{
  "episode_id": "run-17",
  "step": 3,
  "input": {"action": "open"},
  "context": {"difficulty": "hard"},
  "state": {"room": "hall", "has_key": false},
  "state_after": {"room": "hall", "has_key": false},
  "available_actions": [
    {"action": "open"},
    {"action": "search"},
    {"action": "leave"}
  ],
  "output": false,
  "goal": false,
  "terminal": false,
  "source": "rust-blackbox-v3"
}
```

字段说明：

- `episode_id`：独立回合标识。不同回合的历史不会混在一起。
- `step`：回合内顺序。
- `state`：执行动作前的可见状态。
- `state_after`：执行动作后的状态；用于建立转移图。
- `available_actions`：该状态的合法动作全集；用于证明动作覆盖完整。
- `goal`：本次转移是否到达目标。
- `terminal`：转移后的状态是否明确不可继续。
- `history`：可选显式历史；存在时优先使用。

## 主要 API

- `GET /api/scenarios`
- `POST /api/sessions`
- `POST /api/sessions/{id}/probe`
- `POST /api/sessions/{id}/observe`
- `POST /api/sessions/{id}/observations/import`
- `POST /api/sessions/{id}/search`
- `GET /api/sessions/{id}/suggest`
- `POST /api/sessions/{id}/closure/analyze`
- `POST /api/sessions/{id}/validate`
- `POST /api/expr/evaluate`

### 闭环分析请求

```bash
curl -X POST http://localhost:8080/api/sessions/SESSION_ID/closure/analyze \
  -H 'content-type: application/json' \
  -d '{
    "max_cases": 5000,
    "top_candidates": 48,
    "accuracy_tolerance": 0.001,
    "history_depth": 1,
    "coverage_threshold": 0.9,
    "goal_mode": "either",
    "auto_search": true,
    "max_depth": 3,
    "beam_width": 180
  }'
```

`goal_mode`：

- `either`：`goal=true` 或 `output=true` 都视为目标。
- `observation_goal`：只认显式 `goal=true`。
- `output_true`：只认 `output=true`。

## 离线分析

不启动 Web 服务也能直接分析 JSON：

```bash
python scripts/analyze_closure.py \
  --scenario examples/deadlock_scenario.json \
  --observations examples/deadlock_observations.json \
  --history-depth 0 \
  --goal-mode observation_goal \
  --output closure-report.json
```

冲突数据示例：

```bash
python scripts/analyze_closure.py \
  --scenario examples/conflict_scenario.json \
  --observations examples/conflict_observations.json
```

## 对接任意语言黑盒

目标程序只需要把调用结果转换成统一观测 JSON。比如 Rust、Python、JavaScript 或远程 HTTP 服务都可以批量提交：

```bash
curl -X POST http://localhost:8080/api/sessions/SESSION_ID/observations/import \
  -H 'content-type: application/json' \
  -d '{
    "observations": [
      {
        "episode_id": "case-1",
        "step": 0,
        "input": {"x": 2, "tag": "a"},
        "context": {},
        "state": {},
        "output": true,
        "source": "external-adapter"
      }
    ]
  }'
```

## Rule IR 示例

```json
{
  "op": "and",
  "args": [
    {
      "op": "ge",
      "left": {"op": "field", "path": "context.age"},
      "right": {"op": "const", "value": 18}
    },
    {
      "op": "eq",
      "left": {"op": "prev", "path": "input.action", "offset": 1},
      "right": {"op": "const", "value": "knock"}
    }
  ]
}
```

## 测试

```bash
pytest -q
```

当前覆盖：

- Rule IR；
- 简单规则恢复；
- Episode 历史隔离；
- 主动查询；
- 可见上下文冲突；
- 部分域保持 `open`；
- 完整有限域观测闭环；
- 终端 SCC 死路确认。

## 重要限制

- 闭环结论只对你声明的字段 domain、历史深度、状态信息和搜索空间成立。
- domain 没写全，系统最多只能证明“当前测试域无分歧”。
- 时间、随机数、并发、网络、数据库、模型版本等外部变量未记录时，仍会形成隐藏上下文。
- 任意程序的完全等价判断一般不可判定；本项目做的是有限行为归纳、反例搜索和状态空间证据分析。

## PG-388 逻辑漏洞本地展示

PG-388 是一个仅供本机展示的前后端逻辑状态实验台：

```powershell
$env:PG388_PYTHON_IMAGE_DIGEST='sha256:<reviewed-python-digest>'
$env:PG388_NODE_BASE_IMAGE='node:20.11.1-alpine3.19@sha256:<reviewed-node-digest>'
docker compose -f docker-compose.pg388.yml -p pg388demo up -d
```

打开 `http://localhost:3000/pg388`。页面包含 28 个案例（18 个 concrete local canary + 10 个补充分类），可运行 `fresh reset → ASK → failure repair → candidate/reference/negative → replay`。后端只接受抽象枚举，不接收账号、凭据、原始请求值、URL 或响应正文；所有训练、payload catalog 和漏洞晋级标记均关闭。

PG-385/PG-386 页面展示的是另一个 fixture-bound 过滤反馈实验，不能把本地 canary 结果解释为任意网址的通用 WAF、XSS 或 SQL 能力。
#   b l a c k b o x a n a l y z e  
 
