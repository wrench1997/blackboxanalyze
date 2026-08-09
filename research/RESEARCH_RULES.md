# SIFT 根因驱动模型改进规则

本项目允许修改模型架构、记忆系统、训练数据、训练目标、优化方法和推理策略。唯一不可降低的是证据标准。任何“改进”必须遵守以下闭环。

## Rule 0：先找根因，后改模型

单个失败样本、排行榜下降或主观解释都不是架构修改依据。必须先把失败归入一个或多个可验证层级：

1. **数据层**：训练域未覆盖、类别失衡、合成模板单一、标签或 Oracle 错误；
2. **表示层**：token 化破坏结构、JS—轨迹—Rule IR 没有正确对齐；
3. **记忆层**：没有写入关键事实、检索错误、记忆过期、上下文被干扰项淹没；
4. **架构层**：容量不足、注意力跨度不足、任务干扰、路由塌缩；
5. **目标层**：loss 或 RL reward 鼓励了投机策略、冗长查询、误报或不可复现解释；
6. **搜索执行层**：Rule IR 表达力、Beam 预算、Oracle 沙箱或反例搜索不足；
7. **评估层**：模板泄漏、指标与研究目标错位、测试集过窄或随机性未控制。

## Rule 1：每次失败必须形成最小可复现实例

保存模型版本、数据指纹、黑盒版本、随机种子、完整 Episode、检索到的记忆、候选规则、最终输出和验证脚本。不能重放的结论只算假设，不算证据。

## Rule 1.1：规则发现必须标注证据模式

每条规则和特征必须标记以下来源之一，不能混写：

- **White-box extraction**：从明文源码、AST、字节码或公开 IR 直接抽取；
- **Gray-box reconciliation**：静态候选与运行轨迹联合验证、冲突消解和补足；
- **Black-box induction**：源码不可见，只通过主动输入、输出、状态与时间序列归纳。

明文机制可以复用成熟解析器，但静态分析结果仍然只是候选语义；动态加载、配置、转义、运行时版本和外部状态都可能让真实行为不同。黑盒模式必须是一等公民：使用候选分歧、边界探针、状态探索和反例重放逐步补足规则。

每条特征必须保存 `provenance`、`confidence`、`evidence_pointers`、`oracle_fingerprint` 和 `coverage_scope`。当明文抽取与运行时行为冲突时，保留两个版本并创建冲突记录，以可重放运行证据作为“实际行为”结论，禁止静默覆盖。

## Rule 1.2：正向 oracle 是二级复核，不是模型标签

模型只能提出候选族和候选 Rule IR；它不能凭置信度直接宣布“找到出口”。跨应用或未知表面必须先经过严格 OOD abstain，再由已授权、源码指纹固定的族特异 oracle 复核。复核至少要求：同一表面的多编码 pair 一致、证据 SHA-256 有效、目标实例新鲜、oracle 信号满足且没有越权副作用。属性回显、文本回显、JSON 回显和响应头回显必须作为不同表面保存；“有 marker”本身不能替代族特异证据。

## Rule 1.3：反事实表面必须进入每轮评估

每个正向表面至少配套一个语义相近但 oracle 不成立的负表面，并在未见过的目标实例、编码变体和采样 seed 上回放。模型-only 接受率、oracle 复核率和误报率分开报告；负表面通过模型猜测但未通过 oracle 时，记录为表示/表面捷径失败，不得写入长期记忆。

## Rule 1.4：逻辑/访问控制必须使用 typed boundary oracle

逻辑、授权和历史重放类迷宫不能把“返回 200”“资源看起来受保护”或单次高置信度当作出口。每个候选必须绑定到族特异的 typed boundary oracle（例如授权边界越权、业务不变量偏移、challenge/session 历史绑定缺失），并同时满足：

1. 同一候选的 plain 与 URL-percent（或等价编码）回放都得到相同 typed 信号；
2. 具备 fresh target、源码/fixture SHA-256、证据 SHA-256 和 `state_mutated=false` 证明；
3. 配套回放正常 200、正常 403、合法管理员/合法边界和历史绑定正确等反事实；
4. 模型只输出候选 family 与 grammar-checked Rule IR，typed oracle 复核前不得晋升记忆；
5. 至少三个独立目标实例、两个 sampling seed 和一个编码留出切分均通过，才允许进入长期记忆；否则保持 `diagnostic_only` 或 `quarantine`。

PG-PK-10 的 `app/logic_access_fixture.py`、`app/logic_access_oracle.py` 和 `app/logic_access_decoder.py` 实现了这一门禁。该轨道只使用本地 loopback、无凭据、无状态写入、无数据库和无外部网络。

## Rule 1.5：共享表示只负责族路由，不负责正向判定

跨 XSS、SQL、逻辑和访问控制共享 encoder 时，输入必须去除 oracle projection、语义 family 标签、原始响应、路由 token 和来源标识，并先把编码变体 canonicalize 到同一几何表示。共享 head 只能输出候选族与 grammar-checked Rule IR；属性 sink、AST 差分、授权边界和历史绑定等正向结论仍由族特异 oracle 给出。每轮必须报告编码留出准确率、未知表面误路由率、pair embedding 距离和必要的 abstain coverage。共享 head 退化时不得放宽 oracle 或直接扩大模型规模。

## Rule 1.6：表面不变 OOD 与 abstain fallback

共享路由的 OOD 参考只能使用经过审计的表面不变维度（传输方法、状态类别、fresh/local/read-only 证明等）；路径/query 几何、content-type、HTML/JSON/XML parser shape、marker 和 probe 语法不能成为跨表面硬拒绝条件。每条 checkpoint 必须保存 OOD 维度白名单及其 provenance。严格 OOD 或低置信度表示“共享 head 不宣布正向结论”，不表示停止安全探索：预算允许时仍可进入已授权的族特异 typed-oracle lane，并分别报告 router-gated recall 与 fallback-oracle recall。只有 typed oracle、编码 pair、fresh target、源码和证据 hash 全部通过，才能产生正向结果。不同端口/variant/seed 若共享同一 fixture source hash，只算同一数据集的 target 实例；长期记忆门不得把它们当作独立数据源。

## Rule 1.7：跨 source hash 的记忆晋升

同一规则在一个 fixture 的多个端口、variant 或 seed 上通过，只能证明 target/采样稳定性，不能证明跨实现泛化。长期 memory promotion 还必须通过独立 fixture source hash 的交叉回放；报告分别保存 dataset identity、source hash、target instance、sampling seed、pair evidence hash 和 fallback/router 路径。任何一个独立 source 的正向 pair 缺失或反事实误报，候选保持 quarantine，并生成 fresh-reset replay queue。推广到 GPT/大模型训练前，先用这个门确认规则 IR 的语义不是某个路径、标签或响应模板的记忆。

## Rule 1.8：共享 head 置信度只校准路由，不校准漏洞结论

温度缩放和 abstain threshold 必须只在冻结 checkpoint 的独立 calibration split 上拟合，并在未见 source/seed 上复核 ECE、precision、coverage、control false-accept 和 fallback-oracle recall。校准后的概率只能决定主动探针的优先级或是否提供弱 belief prior；它不能替代 typed oracle、pair 一致性、fresh reset 或证据 hash。若校准降低误报却使 fallback 探索被截断，保留 abstain 并继续族特异安全探测，不得为了 coverage 放宽正向门。

## Rule 1.9：SQL 族外负样本增广不得泄漏正向通道

当 SQL decoder 在新 transport 上把参数化值或 baseline 误判为 injection 时，只允许先加入该新 source 的安全负通道（例如 value-only、baseline）作为 counterfactual augmentation；新 source 的正向 fragment 必须继续留出。重训后必须同时复跑旧 source 与新 source，报告 decoder pair candidate、oracle pair、fallback pair 和 false-positive ledger。若只有一个 source hash，promotion 仍为 quarantine；负样本增广不能替代独立正向 source。PG-PK-15 的第三个 `/search?q=...` source 还要求保留 pre-fix 误报快照，并在负样本修复后确认 v1/v2 旧 source 回放无回归，三 source hash 且 false-positive ledger 为零才可晋升。

## Rule 1.10：跨族负类必须独立于全局漏洞正类

一个逻辑/访问控制样本即使由其 typed oracle 证明为正例，对 SQL/XSS 等其他族仍是严格负类。跨族 guard 必须在未见过的 route、query 词汇和响应 schema 上统计每个 family head 的误路由；共享 router 只能提供 belief prior，不能把跨族候选写入长期 memory。若 SQL head 在 logic/access surface 上产生 injection candidate，先保存 pre-fix ledger，再只加入已授权旧 source 的跨族负类，重训后复跑 SQL v1/v2/v3 与新族外 source；要求跨族 candidate 为零、族特异 pair oracle 完整、反事实误报为零，才能把该 guard 标为 pass。guard pass 不等于 memory promotion，source hash 数不足时仍保持 quarantine。

PG-PK-18 进一步要求 logic/access 的第三独立 source 也通过同一门：v1/v2/v3 的 source hash、fresh target、三类 typed oracle、plain/url-percent pair 和零反事实误报都要进入 ledger；任一 source 的正向 pair 不完整时先 quarantine，不能用“其他 source 有成功”掩盖族外漏检。

## Rule 1.11：真实 Docker fresh reset 与未知族必须分开验收

真实容器实验必须按 matched control/candidate pair 启动独立 disposable instance；同一容器跨多个 case 的 `fresh_target` 声明一律不算 fresh reset。未知族必须在训练 registry 之外的独立实现上复放，模型输入不得出现 family、route 词汇或 evaluator 字段；typed positive 只能进入 evaluator 侧，模型必须正确 abstain，不能为了 coverage 硬绑定已知 Rule IR。即使 reset、GET/POST、阴性对照、证据哈希和 Trace 门全部通过，仍只生成 evaluation-only Catalog/Trace；在独立训练后复跑、跨数据集/种子/实现和人工审核完成前，不得训练晋升或写入长期 memory。

## Rule 1.12：跨 seed 的全 abstain 不能算能力提升

冻结候选模型的跨 seed/跨 fresh-target 回放必须同时报告确认召回与弃权率。若已知族在完整有界差分和 typed oracle 均为正的情况下全部 abstain，记为 capability failure，而不是安全成功；该批数据仍是 evaluation-only，不能通过调高 OOD/置信度阈值掩盖漏检，也不能写入训练集或长期 memory。只有已知族达到预注册召回、阴性对照零误报、未知族正确弃权和跨 seed 稳定后，才允许进入下一轮候选训练。

## Rule 8.1：共享模型失败先做实验/工程双路径诊断

共享表示或训练流水线出现回归时，先分别检查：小规模是否复现、seed/族外留出是否退化、指标/Oracle 是否改变（实验路径）；以及数据 hash/血缘、checkpoint、OOM/超时、吞吐、非确定性（工程路径）。两条路径分别验收；混合失败建立两项独立修复，`inconclusive` 先补最小复现。PG-PK-11 的 `app/experiment_engineering_triage.py` 固化此分类，未通过时保持 `diagnostic_only` 或 `quarantine`。

## Rule 2：用反事实消融区分根因

至少比较当前系统与一个针对根因的对照：

- 增加数据覆盖但不改模型；
- 延长原始上下文但关闭外部记忆；
- 保持上下文不变，只注入正确检索结果；
- Dense 与 MoE、共享头与独立头；
- 固定策略与 RL 查询/记忆策略；
- 移除游戏规则抽象或可执行 Rule IR；
- 随机切分与完整漏洞族留出。

没有消融就不能宣称发现了深层原因。

## Rule 3：优先采用最小有效改动

修改优先级为：修复数据/验证错误 → 调整课程或采样 → 调整 loss/reward → 修改记忆策略 → 添加轻量 adapter/head → 扩大上下文 → 扩大 Dense 模型 → 引入 MoE 或新架构。

只有较小改动在受控实验中失败，且架构假设获得证据支持时，才允许显著增加参数或计算量。

## Rule 4：修改前预注册可证伪预测

每个提案必须写明：目标根因、改动、预期改善的指标、不得退化的指标、资源预算和失败条件。例如：

```text
若失败由过期记忆引起，则加入版本指纹过滤后，stale-memory error
应下降至少 30%，Counterexample@10 不得下降超过 1%，检索 token 不得增加超过 10%。
```

## Rule 5：必须在规则族外验证

改动不能只修好原失败模板。至少在完整留出的漏洞族、不同历史长度、不同代码表面形式和不同随机种子上验证。报告均值、方差、误报率、ECE、查询数和 token/计算成本。

## Rule 6：保留、回滚与沉淀

只有满足预注册成功条件且没有触发退化门槛的改动才能进入主线。否则自动回滚。成功实验要沉淀为：新数据生成规则、模型配置差异、训练日志、评估报告和适用边界，写入版本化语义记忆。

## Rule 7：实验成熟后才能进入工程化扩展

大数据、大算力、分布式训练和生产级数据管线不是实验假设的替代品。只有同时满足以下条件，实验才从 `research_only` 晋级为 `ready_for_engineering_scale`：

- 同一结论能够复现，且至少覆盖 3 个独立随机种子；
- 至少完成一次完整规则族留出验证；
- 反事实消融支持所声称的机制，而不只是相关性提升；
- 达到预注册目标，且误报、校准、成本等保护指标没有退化；
- 数据、Oracle、模型、代码和环境具有完整版本指纹与血缘记录。

未通过门禁时继续做小规模、可证伪的研究实验，不得用扩大数据量或模型规模掩盖机制尚未成立。

## Rule 8：扩展失败必须区分实验问题与工程能力问题

工程化扩展出现异常后，任何模型或训练方案修改之前必须执行双路径诊断：

1. **实验问题**：失败在单机小规模也能复现、对 seed/切分敏感、族外效果消失、指标或 Oracle 定义发生变化；
2. **工程能力问题**：单机正确但分布式失败、数据 hash/血缘不一致、OOM/超时/断点损坏、吞吐或 I/O 退化、流水线存在非确定性；
3. **混合问题**：两类证据同时存在时，分别建立科学修复和工程修复，不允许用一个改动同时宣称解决两类根因；
4. **证据不足**：分类为 `inconclusive`，先补最小复现、单机—分布式对照和端到端数据 hash，再决定改什么。

不得通过修改科学假设掩盖工程故障，也不得通过堆基础设施掩盖失败的实验。两条路径必须使用独立验收指标，并将诊断证据写入长期记忆。

## 强制研究循环

```text
Failure → Reproduce → Root-cause taxonomy → Counterfactual ablation
→ Minimal proposal → Pre-registered prediction → Family holdout test
→ Research maturity gate → Experiment / engineering triage
→ Keep / Roll back → Consolidate evidence memory
```

模型的目标不是永远给出答案，而是能识别自己失败的原因，并用成本受控、可证伪的实验推动下一版本。
