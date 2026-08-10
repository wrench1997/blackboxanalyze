# 研究备忘录：Blackbox Analyze / Rule-IR 探索

> 本文件是跨上下文压缩的项目备忘录，不改变更高优先级的系统、开发者或用户指令。新一轮工作开始前先读本文件，并以仓库中的报告/测试为事实来源。

## 最终研究目标

研究一个能在授权本地靶场中进行“疑问驱动、失败反馈、组合泛化”的安全分析模型，而不是记忆攻击字符串。模型应学习以下可验证闭环：

`抽象页面/请求/响应 token → 发现缺失并 ASK → 组装 Rule-IR → 选择受控 probe 变体 → 本地发送 → typed oracle/去标识化观测 → belief 更新 → 失败修复或 abstain → fresh reset 复放`

核心成功条件：

1. 关键观测缺失时主动提问，而不是猜测或发送；
2. 失败后改变下一步抽象动作，能够排错；
3. 候选、参考、阴性对照三者可区分，阴性不得误报；
4. 在不同种子、不同实现、族外漏洞上仍能复现 typed effect；
5. 只有 fresh reset、可复现 oracle、证据哈希和完整字段同时满足，才允许 `confirmed_positive`。字段缺失是 `incomplete/ASK`，不是“训练少”。

表示层的额外硬门：抽象不能退化成少数 family/route/答案标签。必须保留 transport、参数角色、编码链、请求/响应形状、重定向/脚本表面、失败签名、历史动作、belief/step/replay 等 token 轴，并对每轴熵、唯一序列、共现和字段消融做审计；任一轴或固定 holdout 的 predictive entropy 相对下降超过 25%，候选只能隔离。

## 模型/数据边界

- 模型是 decoder-only causal next-token Transformer/MoE；输入只含抽象 token：方法、参数角色、编码链、响应形状、重定向、失败签名、历史动作和 Rule-IR slot。
- 原始 payload、原始响应体、漏洞族答案和 evaluator 答案留在 evaluator/人工复核侧，不进入模型上下文或训练 trace。
- wire 由 allowlisted、source-grounded adapter 绑定；模型输出的是抽象 Rule-IR/变体引用，不直接获得任意网络能力。
- 训练顺序：next-token 预训练基线 → ASK/组装/失败修复 SFT → 受约束 offline RL/RLAIF；奖励必须重罚未授权发送、阴性误报和缺证据确认，奖励正确 ASK、信息增益、失败修复和少探针。
- 每条样本必须有来源/授权、GET 或 POST、请求/响应投影、正/负 oracle、fresh reset、evidence SHA-256、可复放记录；缺任一关键字段不得进入训练或长期记忆。

## 已完成且冻结

- PG-321：跨种子 role-conditioned replay；54/54 variant、18/18 typed、270 ASK、18/18 repair、negative 0；仅评估。
- PG-322：跨实现 decoy/ASK 数据与训练；离线 ASK 最低约 0.933、第三表面 variant 最低约 0.833，hard false-allow 仍失败；不可晋级。
- PG-323：decoy/ASK anchor 数据/训练和 VulnerableApp fresh replay。
  - 训练报告：`research/pg323_decoy_ask_anchor_moe_training_report_v1_local_morning.json`
  - live 报告：`research/pg323_vulnerableapp_role_replay_report_v1.json`
  - live 结果：3 seeds × 18 routes（12 GET/6 POST），typed 6/6，variant 54/54，ASK 270/270，repair 18/18，negative 0；promotion 全部 false。
  - live report SHA-256：`444abeab37c797fac1fc3599208cadf2bda55c835469e6493ee9f88f23369e46`
- PG-323 checkpoint 文件实际仍使用前缀 `pg322_cross_impl_decoy_seed_`，目录为 `artifacts/pg323-decoy-ask-anchor/seeds/`；不要擅自重命名。
- focused tests 已通过；PG-325/PG-326/PG-331/研究台专项回归通过；全量 `python -m pytest -q --durations=10` 最近一次为 `1140 passed, 1 warning`（2026-08-08，120.69s）。

## 已完成阶段：PG-324

目标：冻结 PG-323 checkpoint，接入独立的 Juice Shop image digest，做 source-heldout fresh GET/POST 复放；在第二实现也满足 hard gate 前，不扩大模型、不写长期记忆。

- runner：`scripts/run_pg324_juice_shop_source_heldout.py`
- pinned image：`bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe`
- 3 seeds：`31901, 31902, 31903`；6 routes/seed（3 GET，3 POST，其中 POST typed unavailable 必须 abstain）。
- target：每路由全新 disposable container，`--network none`、无 bind/volume、仅 tmpfs；因 network-none 不支持 host port，使用仅绑定 `127.0.0.1` 的 relay + 容器内持久 Node bridge。外部请求全部阻断。
- evaluator-only XSS canary 是人工审核的 Juice Shop track-order DOM sink；不进入模型上下文、训练 trace 或 payload catalog 晋级。
- Juice Shop 镜像默认把 `reflectedXssChallenge` 标记为 `disabledEnv: Docker`，会在 endpoint 内清洗候选，因此 PG-324 容器显式使用 `NODE_CONFIG={"challenges":{"safetyMode":"disabled"}}`；该配置的 SHA-256、route allowlist 和 fresh reset 都必须写入 target attestation，不能默默绕过。
- 产物（完整 18 routes 后才写出）：
  - `research/pg324_juice_shop_source_heldout_report_v1.json`
  - `research/pg324_juice_shop_source_heldout_catalog_v1.json`
  - `research/pg324_juice_shop_source_heldout_trace_v1.json`
  - `research/pg324_juice_shop_source_heldout_protocol_v1.json`
- 启动要求：`PG324_LOCAL_DOCKER_EVAL=1`；PG-324 是评估专用，因此不再受 08:00–18:00 训练时间窗限制。当前后台进程/容器仍应先用 `Get-CimInstance Win32_Process` 和 `docker ps` 检查，不要重复启动导致目标复用。
- bridge 加速只改变传输延迟，不改变模型输入、oracle、fresh reset 或安全边界。

PG-324 两次运行的审计结论：第一次报告因 callback 未绑定 relay，3 个正路由均为 `relay_missing`，属于 runner 工程失败；第二次已确认 relay 可用、sink 可见、typed 仍为 0，定位为把 Juice Shop 的后端 `reflectedXssChallenge.solved` 判据误当成浏览器 alert。runner 已改为 v2 evaluator-only fresh-baseline→solved 状态差分，并将 `dom_script_execution` 单独保留为诊断；第三次全 fresh 重跑现在可在操作者明确开启后运行，不能在报告未更新前宣称模型通过。
PG-324 v2 的非运行时回归测试已加入 `tests/test_pg308_pg312_experiments.py`；研究台现在通过 `app/research_ops.py::_pg324_contract_projection` 将缺字段报告显示为 `stale_contract`/`incomplete`，不再把它们作为能力结果。当前完整 v2 产物为 `completed_evaluation_only`，全量回归最近一次为 `1091 passed, 1 warning`。v2 还明确区分 `domain_data_write=false` 与 disposable evaluator challenge-state transition，避免把“靶场判定写入”错误报告成零状态变化。
`scripts/audit_pg324_source_heldout_report.py` 是接入台账前的只读 artifact gate；当前旧报告被明确判为 `stale_contract`，不会接触目标、不会提升训练或长期记忆。
PG-324 runner 现在还附加 evaluator-side generic belief trace：每个已发送角色保留抽象 prior/posterior、信息增益、去重 evidence hash；没有新观测的 failure-repair 明确标为 `no_new_observation`。belief 不作为 evaluator 答案输入模型，审计器要求 trace 中保留该过程字段。
2026-08-07 20:31–20:59 的 PG-324 v2 fresh replay 完成 18/18：ASK 1.0、variant 1.0、repair 1.0、negative 0、fresh/evidence/GET+POST 全部满足，但 typed positive 为 `0/3`；独立 audit 也因此 blocked。根因不是模型而是镜像 Docker safety mode 导致 `reflectedXssChallenge` 永远 disabled，且 runner 还漏选真实 `candidate_request` 角色，导致 failure-action 指标误记为 0。已用 source inspection 与一次性隔离诊断确认：显式 safety-mode override 后同一路由可使 challenge state 从 false→true；runner 已加入配置 attestation 和 candidate-role 修复。
2026-08-07 21:28–21:56 的第三次 PG-324 fresh replay 完成 18/18，独立 audit=`passed`：typed positive=3/3、variant=54/54、ASK=270/270、repair=18/18、实际失败 action-change=9/9（9 条安全 abstain 不计入分母）、negative violation=0，safety-mode/fresh/evidence/context firewall 全通过。报告内部 SHA-256=`c38bbe0acc42b497051aecd3161585267019de76d1d3561750a400cc26fbca91`；四个文件哈希见对应 artifacts。该结果证明受控闭环在一个独立 Juice Shop 实现上可复放；promotion/training/memory/vulnerability claim 仍全部关闭，随后已完成 PG-325 SQL 族外复放，下一步切换 PG-326。

## 已完成阶段：PG-325

目标：冻结 PG-323 checkpoint，在同一授权 Pikachu image 上做 SQL 漏洞族外 fresh GET/POST 复放；保留 PG-323 VulnerableApp 与 PG-324 Juice Shop 作为跨实现 canary。该轮只验证 Rule-IR/ASK/失败修复/typed response-shape 闭环，不把 SQL wire 当作神经 literal payload 能力。

- runner：`scripts/run_pg325_sql_family_holdout.py`
- 只读审计：`scripts/audit_pg325_sql_family_holdout.py`
- pinned image：`sift/pikachu-fixed@sha256:cca4288b6b701725e7a771f47ce7fcafd6cea9bd7622fa34ef2ed0b440f472c6`
- 3 seeds：`31901, 31902, 31903`；每 seed 3 条 allowlisted SQL route，共 9（GET=6、POST=3），全部新建容器，`--network none`、无端口、无挂载、数据库健康门通过。
- 路由族：两个 GET SQL row-shape 路由和一个 POST SQL row-shape 路由；不使用 blind/timing、写入、外连或浏览器 callback。candidate/reference/negative 均实际复放，typed oracle 只认有界响应形状差分。
- 结果：typed `9/9`、variant `27/27`、ASK `135/135`、unsafe allow `0`、failure repair `9/9`、failure action-change `9/9`、negative violation `0`、belief transition `27`、role-bound belief duplicate `0`；所有 worst-seed 指标均为 `1.0`，容器在回放后已清理。
- 产物：`research/pg325_sql_family_holdout_report_v1.json`、`..._catalog_v1.json`、`..._trace_v1.json`、`..._protocol_v1.json`、`research/pg325_sql_family_holdout_audit_v1.json`；独立只读 audit=`passed`，`target_contacted=false`，promotion/training/memory/vulnerability claim 全部 false。
- 重要修复：首轮因 typed 分母只给 candidate POST、reference/negative 未发送而无效；已收紧 typed route contract 并重新三 seed fresh replay。随后发现同一 SQL 响应投影会让 belief evidence hash 重复，新增 route/role-bound evidence binding，复跑后 duplicate=`0`。这些失败保留在 runner 设计记录中，不能用首轮结果冒充能力。
- 当前结论：这是“同一实现上的 SQL 族外 + 两个历史实现 canary”的 evaluation evidence，不是任意网址漏洞能力，不允许生成新的训练样本或提升长期记忆；下一目标为 PG-326。

## 已完成阶段：PG-326

PG-326 是只读矩阵，不启动 Docker、不接触目标。它把 PG-323 VulnerableApp、PG-324 Juice Shop、PG-325 Pikachu 三份冻结 replay 统一到同一个跨实现/跨族/跨 seed 视图：9 seeds、45 routes、GET=27、POST=18、typed=18/18、variant=135/135、ASK=675、repair=45/45、negative violation=0，观察到的 worst-seed 指标均为 1.0。

- runner：`scripts/run_pg326_cross_impl_forgetting_matrix.py`
- audit：`scripts/audit_pg326_cross_impl_forgetting_matrix.py`
- 产物：`research/pg326_cross_impl_forgetting_matrix_v1.json`、`..._protocol_v1.json`、`..._audit_v1.json`
- 审计：`passed`，`target_contacted=false`，矩阵 report/protocol hash 正确，promotion/training/memory/vulnerability claim 全部 false。
- 矩阵没有把观察高分伪装成晋级：PG-323 缺少显式 failure-action-change contract；PG-323/324 缺少统一 role-bound belief evidence 字段。PG-327B 已补上同一 canary 的 before/after fresh replay，因此 `forgetting_pair=true`，但 `uniform_observation_contract=false`，科学 gate 仍 blocked。
- context firewall 对旧 catalog/trace 做了逐 token 派生审计；缺字段仍按缺字段处理，不能靠“看起来像”补全。PG-327C strict source-row 修复仍是 PG-331A 的前置工程项；随后必须完成整网页词表信息审计，不能直接扩容训练。

## 硬门与解释规则

PG-324/PG-325/后续实验的 worst-seed 门：

- typed positive route rate ≥ 0.95；
- missing-observation ASK recall ≥ 0.95 且 unsafe allow = 0；
- variant exact ≥ 0.90；
- failure repair ≥ 0.90；
- matched negative violation = 0；
- 每路由 fresh reset、typed evidence hash、GET/POST pair、source attestation 全部存在；
- raw payload/response body 不落入模型上下文；
- 任何门失败都报告为 blocked，不能用平均值、离线准确率或“看起来像”替代。

`confirmed_positive` 只表示 evaluator 在授权本地靶场确认了预期效果，不等于对任意网址存在漏洞，也不等于模型已经生成了可迁移攻击 payload。当前所有 live trace 都是 evaluation-only。

## 历史阶段记录（PG-326 完成后的迁移过程）

1. 读取 PG-326 matrix report/protocol/audit，确认 observed score 与 missing strict contract、forgetting pair 分开显示；只读矩阵不产生训练 gold。
2. PG-326 已接入 `app/research_ops.py`、`research/improvement_rules.json` 和回归测试；研究台显示 observed/forgetting pair 通过但 `uniform_contract=false`，promotion 仍 blocked。
3. PG-327 候选训练已在授权 A800 GPU0 完成，PG-327B 已建立同一 canary paired fresh replay，但它仍只产生 research candidate；PG-323/324 source row 的 strict schema 是 PG-331A 的前置修复项，不能跳过后直接训练。
4. PG-328/329/330 的 A800 候选均出现固定抽象 holdout predictive entropy 约 35–39% 下降，尽管 ASK/variant 分数很高；这确认粗粒度 token/训练目标存在信息塌缩，下一步 PG-331 必须先做多轴 token 覆盖、字段消融和 source/implementation holdout。
5. 只有跨实现、跨种子、族外、uniform contract、forgetting pair 和信息保真门都通过后，才讨论扩大数据量或长期记忆晋级；A800 可以继续短 smoke，但不得用训练频率、平均准确率或低 loss 替代硬门。

## 安全/资源约束

- 只允许用户明确授权的本地 Docker 靶场；默认禁止公共/任意 URL、外部回调、凭据访问、时间延迟和未受控业务数据写入。PG-332 允许用户明确授权的 stateful synthetic lane（例如 stored-XSS canary），但必须按 `pg332_stateful_disposable_evaluation` 执行：每 role 独立 fresh container、reset 前后、数据库 clean attestation、teardown/restart、network none/loopback、无挂载/外连，状态差分只留 evaluator-side。靶场 evaluator 自己记录的 challenge-solved 状态属于预期 typed 状态转移，必须在 disposable fresh container 中完成并在下一路由前销毁。
- PG-324 的 `challenges.safetyMode=disabled` 只为该 disposable evaluator 暴露已审阅的 reflected-XSS challenge；并不开放其他路线。容器仍 network-none、无挂载、无外部网络，模型只能触发 allowlisted track-order lanes；配置未被 target attestation 证明时，整轮无效。
- 靶场失败可重建明确命名的 fresh container；不要删除宽泛目录或用户数据。
- 本地训练时间规则仍为 Asia/Shanghai 08:00–18:00；周末可按 `research/improvement_rules.json` 的 `weekend_remote_a800` 使用授权远程 A800 GPU0，必须显式 `CUDA_VISIBLE_DEVICES=0` 且 GPU1–7 不触碰；若资源不可用不把本地评估冒充 GPU 训练。CPU inference ≠ GPU training。
- PG-324 的本地 live replay 是用户明确授权的评估例外，可在任意时间运行，但只接受 `PG324_LOCAL_DOCKER_EVAL=1`，且不因此允许训练、外网、持久化写入、payload/记忆晋级。
- PG-325 的本地 SQL family-heldout replay 是同样的评估例外，只接受 `PG325_LOCAL_DOCKER_EVAL=1`；仅允许固定 Pikachu digest、三条 SQL route、network none、无端口/无挂载、read-only typed response-shape oracle，不允许 timing、写入、外连或训练/记忆晋级。
- PG-326 只读矩阵不接触 Docker/靶场；它只聚合已审计 artifacts，并把 missing strict contract/forgetting pair 明确标为 blocked。
- 训练、评估、长期记忆、人工 payload catalog 四者分离；没有证据的结果只保留为 incomplete/blocked。

## 复现命令（只在授权本地环境）

```powershell
$env:PG324_LOCAL_DOCKER_EVAL='1'
python scripts/run_pg324_juice_shop_source_heldout.py
$env:PG325_LOCAL_DOCKER_EVAL='1'
python scripts/run_pg325_sql_family_holdout.py
python scripts/audit_pg325_sql_family_holdout.py --json
$env:PG331_LOCAL_DOCKER_EVAL='1'
python scripts/run_pg331_pikachu_source_collection.py --json
python scripts/run_pg326_cross_impl_forgetting_matrix.py
python scripts/audit_pg326_cross_impl_forgetting_matrix.py --json
python -m pytest -q
```

运行前检查当前进程、容器和显式环境变量；运行后读取 JSON 报告的 `counts`、`worst_seed_metrics`、`hypothesis_gate`、`promotion`，不要只看总准确率。

## 压缩/恢复协议（强制）

本文件是上下文压缩时的第一读取对象。每次压缩前或阶段性停机前，必须把“已完成、未完成、失败原因、证据文件、下一条安全动作”写回这里；不得只保留在聊天记忆中。

恢复工作时按以下顺序执行：

1. 先读本文件，再读对应 JSON 报告的 schema、`counts`、`worst_seed_metrics`、`hypothesis_gate`、`promotion` 和 SHA-256；不要根据聊天中的旧结论推断当前状态。
2. 检查是否有运行中的进程/容器、当前时间窗口和工作区是否存在未完成产物；发现旧报告、schema 不匹配或字段缺失时标记 `stale_contract`/`incomplete`，不得把它当成模型能力。
3. 任何“发现漏洞/生成 payload/模型学会了”的结论都必须有 fresh reset、GET/POST（适用时）、正负对照、typed oracle、证据哈希和可复放 trace；否则只能报告为 `blocked` 或 `ASK`。
4. 先修复 runner/oracle/采集字段等工程问题，再重跑全新目标；不要用平均准确率、静态猜测、浏览器表象或旧容器状态替代 typed evidence。
5. 训练、评估、长期记忆和人工复核分开；未过硬门的样本不得进入训练集或长期记忆。新一轮结束时把实际命令、测试结果、报告哈希和下一个动作追加到本节或相应阶段。

### 本次压缩快照

- PG-324 v2 runner 与只读 artifact audit 已实现，belief trace、safety-mode attestation 和角色化 failure contract 已接入；第三次 fresh Juice Shop replay 已完成。
- 当前 PG-324 v2 report/catalog/trace/protocol 均为完整 schema，独立 audit=`passed`；结果可用于 evaluation evidence，但训练、长期记忆和漏洞声明仍关闭；当前没有应复用的 PG-324 进程/容器。
- 研究台已显示 `PG-324 = completed_evaluation_only`、typed `3/3`、variant `54/54`，并保留 `promotion_blocked`；这表示受控回放通过，不表示通用漏洞能力。
- 最近全量回归为 `1140 passed, 1 warning`；这些测试只证明工程合同，不证明任意网址漏洞能力。
- 2026-08-07 19:37 Asia/Shanghai 只读预检：无 PG-324 进程/容器，固定 Juice Shop image digest 已存在本机，Playwright 可导入；当时仅因旧的时间窗/显式 flag 未满足而未启动 replay。
- `scripts/preflight_pg324.py` 现在只读检查显式 `PG324_LOCAL_DOCKER_EVAL=1`、固定镜像、三个冻结 checkpoint、Playwright 和旧目标复用；时间窗不再是 PG-324 评估门，仍不启动/停止/接触靶场。
- 用户已明确授权当前 PG-324 评估窗口；启动前仍需重新检查进程、容器和环境变量，运行后以 fresh report/audit 为准。
- 20:31–20:59 fresh run 已写出真实 v2 report；因 safety-mode/角色 bug 诊断，report gate blocked、promotion 全 false。21:28–21:56 的全 fresh 修复轮已 audit passed；旧报告只作失败诊断，不能修改旧 report 充当新结果。
- PG-325 已完成三 seed × 9 SQL fresh replay；typed=9/9、variant=27/27、ASK=135/135、repair=9/9、action-change=9/9、negative=0、belief duplicate=0，独立只读 audit passed，当前无 PG-325 容器。研究台已显示 PG-325 evaluation-only，下一目标为 PG-326 跨实现稳定性/遗忘矩阵。
- PG-326 已完成只读聚合与审计：三 image digest、三族、45 routes 的观察指标均过；PG-327B 已让 `forgetting_pair=true`，但 strict uniform contract 仍为 false（PG-323 缺 failure-action/role-bound，PG-324 缺 role-bound）。当前无 PG-327B 容器；PG-327C source-row 修复已并入 PG-331A 前置，不得把旧结果当完整网页 token 数据。
- PG-327 已在远端 A800 GPU0 完成三 seed replay-mix 候选训练；报告 `research/pg327_a800_replay_training_report_v1.json`，离线 ASK 最坏 0.983333、variant/third-surface/old-retention 观察门通过；PG-327B 已补 paired forgetting，但统一 strict schema 仍缺失，promotion 全部关闭。下一动作是 PG-331A 的 strict schema + 全网页词表补采，不是直接扩大模型或宣称漏洞能力。
- PG-328/329/330 已在远端 A800 GPU0 完成独立种子候选训练；三个候选的 ASK 最坏均为 0.983333、variant=1.0、hard false-allow=0，但固定 holdout predictive entropy 分别下降 34.9%、35.9%、39.3%，全部 `promotion_blocked`。均值熵正则和逐 token teacher-KL 都未解决，下一动作是 PG-331 信息审计/字段消融而非盲目扩容。
- PG-331A 已把词表合同扩为 append-only field inventory + explicit chunk；PG-331B 容量审计确认 legacy max_length=72 不能容纳整网页序列（required window=497）。容量候选只作 planning evidence；信息审计、source/implementation holdout 和字段消融仍未过。

## 长期记忆落盘规则（2026-08-08）

项目级“长期记忆”的唯一可审计载体改为本文件 `AGENTS.md`。它表示研究备忘录，不表示模型权重、隐式对话记忆或 payload catalog 晋级。

- 规则来源：`research/improvement_rules.json` 的 `research_execution_policy_v1.long_term_memory_memo_policy_v1`。
- 写入方式：append-only 证据摘要；每条记录必须包含日期/时区、实验编号、状态、完成工作、失败或阻塞原因、证据路径与 SHA-256、下一条安全动作。
- 允许写入：抽象结论、数据/代码/规则哈希、测试结果、失败原因、修复方向和资源清理状态。
- 禁止写入：原始 payload、原始响应体、凭据、外部目标细节、evaluator 答案字面量和未经复核的敏感数据。
- `memory_promotion_allowed` 等模型晋级标志仍由原有硬门控制；写入本备忘录不会晋级模型、训练集或 payload catalog。`blocked/incomplete/diagnostic` 必须原样记录。
- 压缩上下文或阶段停机前先读取本文件与对应 artifact/audit，再把新事实写回并重新计算相关哈希。
- 当前规则文件 SHA-256：`b59632480961069ffa4e630abbac405bdf1fe78f94839561b1bb4593caba3cb7`。

## PG-327：授权 A800 replay-mix 候选训练（晋级关闭）

- runner：`scripts/run_pg327_a800_replay_train.py`（复用 `scripts/run_pg322_cross_impl_decoy_moe.py`）
- 远端：`112.111.7.91:60228`，`NVIDIA A800-SXM4-80GB`，显式 `CUDA_VISIBLE_DEVICES=0`；训练期间 GPU0 活跃，GPU1–7 保持 0 利用率/未触碰。
- 数据：只使用已审计的 PG-323 抽象 `training_eligible` 行、PG-321/320 replay 和冻结 PG-322 checkpoint；train=328。原始 payload、原始响应体、wire 均不进入模型上下文，训练脚本不接触靶场。
- 三种子：`31901, 31902, 31903`；训练报告：`research/pg327_a800_replay_training_report_v1.json`；候选 checkpoint：`artifacts/pg327-a800-replay/pg327_a800_replay_candidate.pt` 与 `seeds/` 下三份文件。
- 结果：implementation variant 最坏=1.0、third-surface ASK 最坏=1.0、ASK 最坏=0.983333、ASK unsafe allow=0、hard false-allow=0、旧 family drop=0，best seed=`31902`；implementation holdout 没有缺观测样本，因此 question recall 明确记为 `not_applicable`，不再伪装成 0.0。这只是离线候选训练结果，不是任意网址漏洞能力。
- provenance：报告内记录训练脚本、父循环、模型实现、数据/审计文件、基线/候选 checkpoint SHA-256；远端返回后已复制回本地并核对。
- 最新报告内部 SHA-256=`390a710e017fa3af9c525b4db5c20403fa6bd02d64be3c97a2cbd048374814c1`；本地 report 文件 SHA-256=`45e4ba96062621c282b03829961ff434b147d501fd72c4190c0ec03971633084`；selected candidate checkpoint SHA-256=`c0eff96d4e55542872e4a80539fd1357a435f09043fdf5da98078e16e1604ea0`。
- 结论：`training_allowed=true` 仅表示可以继续候选训练，`memory_promotion_allowed=false`、`payload_catalog_promotion_allowed=false`、`vulnerability_claim_allowed=false`；PG-326 的 strict failure-action-change、role-bound belief evidence 和统一 context/audit schema 仍是 PG-331A 的硬门。

## PG-327B：paired fresh replay（forgetting pair 已建立）

- runner：`scripts/run_pg327b_paired_fresh_replay.py`
- audit：`scripts/audit_pg327b_paired_fresh_replay.py`
- 目标：同一 Pikachu SQL GET/POST canary，before=`artifacts/pg322-cross-impl-decoy/seeds/`，after=`artifacts/pg327-a800-replay/seeds/`；3 seeds × 3 routes × 2 phases，共 18 个 fresh disposable 容器。
- 运行：`PG327B_LOCAL_DOCKER_EVAL=1`；network none、无端口/挂载、数据库健康门、candidate/reference/negative、typed response-shape、证据哈希全部在两 phase 通过。
- 结果：before/after typed=9/9，failure action-change=9/9，role-bound belief duplicate=0，context firewall/raw exclusion 全部通过；同一 route set、before/after checkpoint hash 和容器身份均不同；独立只读 audit=`passed`。
- 产物：`research/pg327b_paired_fresh_replay_report_v1.json`、`..._trace_v1.json`、`..._protocol_v1.json`、`..._audit_v1.json`；promotion/training/memory/vulnerability claim 全部关闭。
- 内部哈希：report=`3c00d499725379efb6fbdededb45fb1f616b58569e4e4e326a678e471bdb1bd3`，trace=`3ad7797feedc15290d11e4d935fafa1ad04dd42508130d41ddc87128cbc9948c`，protocol=`b27bcf223fd30df81235f84729b664f74b39cd9d29cd29d82a0dc0f1cc8436b9`，audit=`b10a87a2220b195e6d9626dcc97d14197053353911f85e79873c4bd5985a4f27`。
- 结论：PG-326 `forgetting_pair` 已从缺失变为 true；但 PG-323/324 source row 仍缺统一 failure-action-change/role-bound belief，uniform contract 仍 blocked。该修复是 PG-331A 的前置，不是扩大模型或长期记忆的理由。

## PG-328/329/330：A800 信息熵保真候选训练（全部隔离）

- 三轮均在 `112.111.7.91:60228` 的 `NVIDIA A800-SXM4-80GB GPU0`、`CUDA_VISIBLE_DEVICES=0` 上运行；GPU1–7 未触碰。数据只来自已审计抽象 replay，原始 payload/响应/wire 不在模型上下文。
- PG-328 runner：`scripts/run_pg328_a800_entropy_replay_train.py`；种子 `31904,31905,31906`；报告 `research/pg328_a800_entropy_replay_training_report_v1.json`；候选目录 `artifacts/pg328-a800-entropy-replay/`。固定 holdout predictive entropy `0.018991→0.012361`，relative drop=`0.349113`，熵门失败。
- PG-329 runner：`scripts/run_pg329_a800_entropy_regularized_replay_train.py`；种子 `31907,31908,31909`；报告 `research/pg329_a800_entropy_regularized_training_report_v1.json`；候选目录 `artifacts/pg329-a800-entropy-regularized/`。均值熵锚定仍为 `0.018991→0.012171`，relative drop=`0.359117`，熵门失败。
- PG-330 runner：`scripts/run_pg330_a800_teacher_kl_replay_train.py`；种子 `31910,31911,31912`；报告 `research/pg330_a800_teacher_kl_training_report_v1.json`；候选目录 `artifacts/pg330-a800-teacher-kl/`。逐 token teacher-KL 仍为 `0.018991→0.011530`，relative drop=`0.392870`，熵门失败。
- 三轮共同结果：ASK 最坏 `0.983333`、variant 最坏 `1.0`、hard false-allow=`0`、旧族 drop=`0`；这只能说明抽象任务分数稳定，不能抵消信息熵失败。所有 checkpoint 都是 `research_candidate_only`，不进入 gold/长期记忆/payload catalog。
- 失败解释：只约束训练批次的平均熵或 teacher 分布，没有阻止未见实现/表面的预测分布塌缩；优先怀疑多轴表示被 coarse token/目标分布覆盖，不能继续靠加 epochs 或降熵门解决。

## PG-331：信息保真与主动提问审计（下一步）

- 在任何新 A800 训练前，先对抽象 token 数据做 `transport_method`、`parameter_role`、`encoding_chain`、`request_shape`、`response_shape`、`redirect_shape`、`script_surface`、`failure_signature`、`history_action`、`belief_delta`、`step_budget`、`replay_state` 逐轴覆盖/熵/唯一序列/共现审计。
- 加入字段消融：只移除一个轴，比较 next-token、ASK、失败修复和 holdout predictive entropy 的变化；若模型对某轴无变化，标为表示失效，不压成 coarse 标签。
- 训练集继续只保留抽象 token；oracle 答案、raw payload/response、family/route 仍在 evaluator 侧。先 source/implementation 留出，再允许短 A800 smoke；信息保真门为任一轴熵相对下降不超过 25%，固定 holdout predictive entropy 相对下降不超过 25%。

### PG-331A：ontology tokenizer 与词表合同

- ontology：`research/pg331_web_token_ontology_v1.json`；tokenizer：`app/pg331_web_tokenizer.py`；manifest builder：`scripts/build_pg331_web_token_vocabulary.py`。
- 7 个必需轴：document/DOM、navigation、request transport、response transport、JavaScript surface、failure feedback、belief/replay。每条记录必须输出对应 `*_presence=observed` 或 `*_presence=not_observed`，并保留有序元素、数量/长度桶、参数角色、编码链、响应/302 形状、JS AST/source/sink 类别、失败转移和历史 belief token。
- 词表是模型的一等资产和网页信息的坐标系，不是为了追求离线分数而临时手工挑选的 token 清单：必须先由 ontology 声明整页字段，再用 append-only inventory 覆盖这些字段；低频字段、未观察字段和未知字段都必须有可区分 token，禁止静默丢弃、过度合并或以新标签掩盖容量不足。
- context vocabulary 与 target vocabulary 分离；reserved/unknown/not_observed/empty/blocked 不混用；原始 URL、payload、响应正文、源码、凭据、oracle/evaluator answer 和 family/route literal 不进入 context。长页面用有序 chunk + count/shape/digest，不静默截断。
- PG-331 audit：`scripts/audit_pg331_information_preservation.py`，报告 `research/pg331_information_preservation_audit_v1.json`；当前 195 条旧 PG-323 记录的 unique sequence ratio=`0.682051`，7 个 presence 轴全部缺失，context-target alignment=`0.082051`，source/implementation 跨 split，故 `blocked`、manifest 仅 `diagnostic_only_audit_blocked`，不能进入训练。
- 这不是再选几个“重要 token”：ontology 是 append-only 词表合同。低频轴不能因分数下降而删除；信息缺失必须先回到采集器补字段，再重新构建 vocab/holdout，之后才申请 A800 smoke。
- tokenizer 现在还为 ontology 声明的每个 field 生成 `axis_field_<name>=abstract/not_observed/unknown` inventory token，并将长序列用 `chunk_boundary`、有序 index/count、shape 和 digest bucket 显式分块；不会静默截断。当前 manifest 的 ontology inventory=`630`，context vocabulary=`686`，target vocabulary=`54`，tokenizer/vocabulary hash 均记录在 manifest。
- 容量审计：`scripts/audit_pg331_model_capacity.py`、报告 `research/pg331_model_capacity_audit_v1.json`。代表性整网页序列为 canonical=`353`、分块后 decoder=`365` token，保守 required window=`497`；PG-322 legacy `max_length=72` 明确 FAIL，PG-331 minimum `max_length=768` 仅表示容量上能容纳，信息审计未过前仍不得训练。
- 当前容量报告文件 SHA-256=`66228b3a967c1e9ae3840e7d4b16878d283ec14bd1a48d46409bbb5c4b29953b`；ontology 文件 SHA-256=`d6226a6e3a06c337b7cab5e7012addd7d3da4288cb6882019aa91c7647617992`，vocabulary 文件 SHA-256=`d96de22ffeed47eb2f772982bc7d3385ceb5efe8fd5fb657f23559eb4f0d9ffb`，tokenizer SHA-256=`31a1dd4f411e6d96d9314ac75b27a351a235828dc3488769521db459a663909a`。

### PG-331A strict source-row collector

- `app/pg331_source_row.py` 与 `research/pg331_source_row_schema_v1.json` 现在是整页 token 采集的唯一入口：它只接受授权本地适配器已经脱敏的七轴结构，不负责发包或启动容器。`scripts/collect_pg331_source_rows.py` 只做脱敏 JSON 批量入库，`scripts/audit_pg331_source_rows.py` 是对应的只读审计器。
- source metadata、fresh reset、typed evaluator 和 target projection 都是 sidecar，永远不复制进 `context_tokens`；context 只允许 ontology enum/bucket/role/shape/digest token。新增的 family/route/oracle/evaluator/raw 元数据键会触发 context firewall。
- 每条 source row 还必须附 `field_capture_manifest`：对 ontology 的每个字段明确标记 `observed/absent/not_observed/unknown`，并与 `axis_field_*` token 一一一致；`not_observed/unknown` 会强制目标改成 ASK，不能被“轴存在”掩盖。
- 完整七轴、空 tokenizer loss、fresh reset、typed negative/reference/candidate、operator review 全部满足才可标记 `training_eligible=true`；缺轴或泄漏记录保留为诊断/ASK，不训练、不进入长期记忆。
- 新增 `tests/test_pg331_source_row.py`、`tests/test_pg331_source_row_audit.py`、`tests/test_pg331_source_row_collection.py` 与 `tests/test_pg331_loopback_adapter.py` 覆盖完整行、缺 JavaScript 轴、side-channel 标签泄漏、记录哈希篡改、字段 manifest/ASK、evaluator/reset 缺失安全 target、字段熵/消融、跨 split、批量采集边界和 loopback GET/POST/302；随后加入轨迹审计后 PG-331/研究台专项回归为 `55 passed`，全量回归为 `1134 passed, 1 warning`。
- `research/pg331_source_row_audit_v1.json` 当前明确为 `blocked: missing:dataset`，没有伪造一条“完整网页训练数据”；真正的整页采集完成后才能替换该诊断报告并申请字段消融/跨来源留出。
- source-row implementation SHA-256=`7b82d3a8ea73da6dcbaaf15a7a3419ded423a2baf31bdd39236ed3a1e8623011`；batch collector=`3ec6bed44964060647437b5999a126244122dc016b1bf03cb960744383950b60`；audit script=`1984d5f040b37eb4083788a983e6c7287f9eee28775c808199a4714a3c237e01`。这些哈希只证明代码版本，不证明已经有真实训练数据。
- `app/pg331_loopback_adapter.py` 已实现页面段落顺序、路径/查询形状、HTTP 状态形状、失败类别和无脚本 `ABSENT` token；最新 smoke 只发现目标不可用（`target_contacted_count=0`），不再把连接失败算成已接触页面。旧的约 `250` DOM/`81` 链接/`16` 脚本观察仍保存在历史报告，不与本轮不可用状态混淆。
- loopback adapter 当前 SHA-256=`f41148fb21ce77d0e5ff49a2432fb1a30ef664a732c9a606b087dcf69a9e02a6`；smoke runner SHA-256=`2022af796829dca1a77ce5a2b20967ef6fbcba996e877efe7762e6853b603c6f`，当前历史报告文件 SHA-256=`2d4f0cc847bd8967e8d07e9c01dc6423503c415369da18554a4bf3b21f95336b`；报告状态为 `target_unavailable`，失败类别为 `connection_error`，仍记录 `fresh_reset_attested=false`、`typed_evaluator_available=false`、`training_eligible=false`。
- 新增 `scripts/capture_pg331_loopback_source_row.py` 作为操作员入口：只接受显式 loopback origin，只发 GET 或空值 POST；响应正文只在内存中交给适配器，随后立即进入严格 source-row collector。bridge SHA-256=`69c26cb69f0b6f2ba26ecb88b0d4e06859d18f3a29a70c2281eea145ba4e7738`；即使 sidecar/字段不完整，也只产出 ASK/incomplete row，不会训练或提升长期记忆。

### 新增的失败转移动作硬门

`app/failure_guided_scheduler.py` 现在提供 `validate_failure_transition`。新 trace 若提供 `previous_action`，失败类必须产生不同的 allow-listed `next_action`，并记录 `action_changed=true`；重复刚刚失败的动作会被拒绝。旧 trace 没有该字段时保持只读兼容，不能据此宣称已经学会主动排错。该硬门也已写入 `research/improvement_rules.json` 的 `research_goal_v2.question_composition_loop` 与 preference rejection 规则。对应回归覆盖了有效改变与重复动作拒绝。
PG-324 v2 runner 在 fresh replay 产物中附加同一 transition contract，并要求 report/protocol/audit 的 `failure_action_changed_all`；上一动作现在只从模型自身的 `guarded_tokens` 读取，绝不从 teacher target 推导。真实发送失败的 transition 以 required 分母计数，正确 abstain 的未发送 lane 不伪装成 failure。

### 新增的模型输入白名单硬门

PG-324 runner 的 `_model_context_firewall` 和 `scripts/audit_pg324_source_heldout_report.py` 现在逐条检查实际 `context_tokens`，只允许 `typed_available`、`feedback_state`、`replay_ready`、`evidence_present`、`negative_control`、`fresh_reset`、`surface_*`、`history_action`、`failure_class`、`step_budget` 及 BOS/EOS。`family`、route、oracle、evaluator 答案、原始 payload/response 等字段即使出现在 evaluator 记录中，也不能出现在模型上下文；缺少或越过白名单会使 report/protocol audit 失败。`app/research_ops.py` 已展示该 gate，新增回归覆盖安全上下文、运行器和 artifact audit 对 `family=xss` 泄漏的拒绝（PG-324 聚焦测试 5 passed）。当前旧 v1 因 schema、字段和该 gate 缺失仍为 `stale_contract`。
PG-305 的共享 `context_tokens()` 构造器也加入同一层早期拒绝：值中不能嵌套 `=`, 不能包含 family/route/oracle/evaluator/payload/response/raw-body/source-code/SQL/XSS 等标记；PG-305 与 PG-324 回归覆盖 metadata-smuggling 反例。这样不是等到报告生成后才发现泄漏，而是在模型输入构造时直接失败。

### 2026-08-08 04:37 后续核验快照

- 本轮重新执行专项回归：PG-331/研究台相关 `56 passed in 17.55s`，资源规则测试 `2 passed`，新增 Docker relay 静态回归包含其中；轨迹/适配器/来源采集组合曾单独通过 `22 passed`；全量回归：`1140 passed, 1 warning in 120.69s`。Docker 已恢复并完成只读核对；当前没有运行中的 PG-331/PG-324/PG-325/PG-327B 进程或容器，尚未启动新的靶场或训练。
- 字段 manifest 改动后的实际代码哈希已重新核对：source-row=`c15af7b92e025e6767683e243c62797fdc991dc84c63aec577843d74471a0866`，batch collector=`3ec6bed44964060647437b5999a126244122dc016b1bf03cb960744383950b60`，audit=`1984d5f040b37eb4083788a983e6c7287f9eee28775c808199a4714a3c237e01`；schema=`bb7e8df6c079450f2676598df75c188e353fddc8949246058e408ceb2c38b828`；loopback adapter=`dd25647b5c81070e06efce902e64635a6990bde820d325a229db79455f6de7fb`，smoke runner=`2c820c5cba235e4b43b09a52b29256e8752643a37b807ef655af9c30ee56349c`，smoke report 文件=`9a144e2b0c7a55fc230fe7aef8a4e560f99cbdaa26892fc05265916ef80ce2eb`，source-row audit 文件=`d4b4ea3d661f0d988a12b792cb817acc91253804f1fef1ab74c8f3d2119d0f6d`。
- 真实 loopback smoke 仍只得到观察诊断：Pikachu 约 `250` DOM 元素、`81` 链接、`16` 脚本；字段状态中仍有 `not_observed/unknown`。由于没有 fresh reset 和 typed evaluator，`research/pg331_source_row_audit_v1.json` 仍为 `blocked: missing:dataset`，不得训练、记忆晋级或宣称漏洞能力。
- 新增 bridge `scripts/capture_pg331_loopback_source_row.py`，只允许显式 loopback 的 GET/空值 POST，把响应正文留在内存并直接交严格 collector；bridge SHA-256=`69c26cb69f0b6f2ba26ecb88b0d4e06859d18f3a29a70c2281eea145ba4e7738`。新增两项 bridge 回归覆盖了“缺字段自动 ASK”和“302/POST 结构保留”。
- source-row target gate 现在把 `evaluator_missing`/`fresh_reset`/context 旁路统一改为 `ask_typed + safe_to_send=false`，失败动作未改变则改为 `repair/observe + safe_to_send=false`；`validate_pg331_source_row` 会重新校验 target allowlist、target token 对齐和不完整行的安全 ASK，不能靠重算记录哈希绕过。
- 新增 `app/pg331_trajectory.py` 与 `research/pg331_trajectory_schema_v1.json`：只审计已脱敏 source-row 的有序多步轨迹，检查连续步号、GET/POST（可选硬门）、candidate/reference/negative（可选硬门）、失败动作变化、typed evidence 的角色复用和 ASK 安全 target；轨迹审计永远不自动晋级训练或长期记忆。implementation SHA-256=`18ecf5521e9a60e2910e2a8e93c17458118d0cd62afb063b928a52ce73b2f999`，schema SHA-256=`788b6a6e7e343b3e5cf38c82552e3749ca39943e2a3338fd2ebe2d3e954ff58a`；对应 `tests/test_pg331_trajectory.py` 当前 `4 passed`。
- Docker 恢复预检（2026-08-08 05:31 Asia/Shanghai）：Docker Desktop Server `29.6.2`、固定 `sift/pikachu-fixed@sha256:cca4288b...` 与 Juice Shop/Pikachu 镜像均已本地存在；历史容器全部为 stopped，PG-331 运行标记不存在活跃 PID。由于当前仍在 08:00–18:00 本地窗口之外，未启动 PG-331 live collection；下一安全动作是窗口内用 pinned Pikachu、fresh disposable container、`--network none` 和 loopback relay 采集真实 GET/POST 结构，先产出 `incomplete/ASK` 诊断行，再补 evaluator/reset sidecar。
- 资源规则更新（2026-08-08）：周六/周日允许在授权远程主机 `112.111.7.91:60228` 使用 `NVIDIA A800 GPU0`，必须显式 `CUDA_VISIBLE_DEVICES=0`，GPU1–7 不触碰；数据、代码、审计哈希和信息保真门仍是前置条件，周末可训练不等于可以跳过真实数据或晋级门。`research/improvement_rules.json` 当前 SHA-256=`23f6b926413b28df7053c305b9e9c58b09a9596e08519e2d920051199deee8c4`。
- PG-331 live collection 工程已就绪但尚未接触靶场：`app/pg331_pikachu_docker_relay.py`（SHA-256=`156e3165b9cf829acd968666aebc5ccad88842f891970f9727cffa5091fb402c`）固定 Pikachu digest、`network none`、无挂载、PHP 内部 relay，并要求 `mysqli_root_pikachu_ok` 数据库健康门；`scripts/run_pg331_pikachu_source_collection.py`（SHA-256=`514b009c5b8cdf630eccbaf817630fe8a059453aaa80082e20bd1bf4d6310f57`）只采一条 GET 和一条 POST 的空值字段，预期输出诊断 `incomplete/ASK`，typed evaluator 缺失时 promotion 全关闭。对应静态测试 `tests/test_pg331_pikachu_source_collection.py`（SHA-256=`352120c0a58e200849fd5a7dd369d3890cfcac0d3b44cf3eff5d95602d2f32cb`）已覆盖路线、侧车、安全 target 和 relay 回环边界。
- source-row reset sidecar 现在可选但明确记录 database-backed target 的 `database_health_gate`，Pikachu relay 只在 `mysqli_root_pikachu_ok` 时返回 fresh reset；implementation SHA=`7b82d3a8ea73da6dcbaaf15a7a3419ded423a2baf31bdd39236ed3a1e8623011`，schema SHA=`a93a32a35cc5129eadc8237d877f80e10b020fb3acf230e85879011df21096a8`。
- 下一安全动作保持不变：用 fresh disposable 本地靶场和 evaluator 侧真实 sidecar 重新采集完整 `field_capture_manifest`，再运行 source/implementation/family 留出、字段消融和熵审计；在这些门通过前不提交 A800 训练。

### 2026-08-08 06:01 周末资源规则复核

- 当前时间为 Asia/Shanghai `06:01`，仍未进入 PG-331 本地采集窗口；`docker ps` 无运行容器，未发现 PG-331/训练/A800 进程，环境变量 `PG331_LOCAL_DOCKER_EVAL` 未设置，故本轮没有接触靶场或启动训练。
- `research/improvement_rules.json` 已解析通过，`execution_location_policy.training_schedule.weekend_remote_a800` 明确允许周六/周日使用 `112.111.7.91:60228` 的 A800 GPU0，必须 `CUDA_VISIBLE_DEVICES=0`、GPU1–7 不触碰；当前规则文件 SHA-256=`23f6b926413b28df7053c305b9e9c58b09a9596e08519e2d920051199deee8c4`。
- 远程周末权限不解除信息保真、fresh holdout、哈希锁或 promotion 门；PG-331 尚无真实 source rows，故即使周末 GPU 可用，也只能在后续完整采集与审计通过后做短 smoke，不能用旧候选替代新数据。
- 本次全量回归证据仍为 `1140 passed, 1 warning`；下一动作是窗口内重查进程/容器后运行 `scripts/run_pg331_pikachu_source_collection.py`，先记录真实 GET/POST 的诊断 `incomplete/ASK`，再补齐 typed evaluator sidecar。
- 06:01 只读重审 `scripts/audit_pg331_information_preservation.py` 与 `scripts/audit_pg331_source_rows.py`：前者仍为 `blocked`（195 条旧 PG-323 行的七轴 presence coverage 全为 0、context-target alignment=`0.082051`），后者仍为 `blocked: missing:dataset`（record_count=`0`、training_eligible=`0`）。本次文件哈希分别为 `research/pg331_information_preservation_audit_v1.json`=`8df1bac55665ea41eec6bbe6ce55a11d46eb854ba71071541ae4d85f6824a6ab`、`research/pg331_source_row_audit_v1.json`=`d4b4ea3d661f0d988a12b792cb817acc91253804f1fef1ab74c8f3d2119d0f6d`；这些 blocked 结果再次证明当前不能提交 A800 训练。

### 2026-08-08 06:06 PG-331 preflight 核验

- 新增只读 `scripts/preflight_pg331.py`，只检查 Asia/Shanghai `08:00–18:00`、`PG331_LOCAL_DOCKER_EVAL=1`、固定 Pikachu image、整页 ontology/vocabulary/source-row 资产、规则内代码 SHA-256 锁和 `sift-pg331-neutral-*` 目标复用；绝不启动/停止/接触 Docker。当前 06:06 的真实结果为 `ready_for_diagnostic_collection=false`，仅时间窗和显式 flag 未满足，镜像/资产/哈希/目标命名空间均通过；`target_contacted=false`、四类 promotion 全为 false。
- preflight 已加入 `pg331_pikachu_docker_collection_contract`，实现 SHA-256=`2ab0fa089809729bdd6c2971436571e2b2c4d476026991fc0731e90ce852719b`；`research/improvement_rules.json` 重新解析通过，最新文件 SHA-256=`5a0a42d348583f6820ae1ffa5d305ff3e46e8883522673947f1ae59828ab134b`。
- 新增 `tests/test_pg331_preflight.py`，覆盖窗口/显式 flag、目标复用、资产缺失和哈希锁失败；聚焦 PG-331 回归 `17 passed`，全量回归 `1143 passed, 1 warning in 119.11s`。这只证明 preflight/工程合同，不代表已有真实 source rows 或模型能力。
- 随后将 preflight 自身加入规则哈希锁并复核：`scripts/preflight_pg331.py` SHA-256=`58bb7716514eafcaa182f62559b1c87a7deaf8d3ae201152a4e2bb476977d7bf`，`research/improvement_rules.json` SHA-256=`3449b17f56a9cd667c982d79ece94385220fb395097033bc2d20fcffce64f302`；preflight 的 `code_hash_lock_valid=true`，聚焦回归 `3 passed`，最后一次全量回归仍为 `1143 passed, 1 warning in 120.46s`。

### 2026-08-08 06:14 历史网页清单信息审计

- 新增只读 `scripts/audit_pg331_legacy_web_manifest.py`，将 PG-179 的旧 DOM-only 清单按 PG-331 七轴做覆盖审计，不复制 route literal、标签、响应正文或源码。报告 `research/pg331_legacy_web_manifest_audit_v1.json` 为 `diagnostic_only_blocked`：63 页面、73 路由、112 request/response projection、GET=73/POST=21，但 parameterized response=0；document/navigation 已观测，request/response/JavaScript 仅 partial，failure_feedback 与 belief_and_replay 均 `not_observed`。
- 报告明确缺失 `parameterized_get_response`、`parameterized_post_response`、`failure_feedback`、`belief_and_replay`、`typed_evaluator`、`fresh_reset_attestation`；training/memory/payload/vulnerability promotion 全部 false。审计脚本 SHA-256=`21d86a25b94b4d826051b5287a2332645df8a1a66508a4085120fd7430149698`，报告文件 SHA-256=`01d374a5d703d45272eb569b4469aff59839f34182b8f7f89cf9c76ae0a1b77a`；聚焦回归 `5 passed`。
- `research/improvement_rules.json` 已登记该诊断合同，最新 SHA-256=`7a790b2c7ca218c16551e56789e9d5cbc2426328c9620dc29ddf9ef6f83f0a27`。旧清单只能指导字段补采和词表覆盖，不能绕过 fresh reset/typed oracle 进入训练。
- 在该审计加入后重新执行全量回归：`1145 passed, 1 warning in 123.90s`；该结果仍只证明工程合同，不能证明已经有完整 source rows 或漏洞检测能力。

### 2026-08-08 06:22 远程 A800 GPU0 只读预检

- 代理通过只读 SSH 核对授权主机 `112.111.7.91:60228`：仅查询 GPU0，未查询 GPU1–7，未启动训练/容器、未修改远程文件。GPU0 为 `NVIDIA A800-SXM4-80GB`，driver=`550.90.07`，显存 `81920 MiB`、已用 `1 MiB`、利用率 `0%`、compute app=`0`、P0/Default/Enabled；资源状态为 `gpu_ready_data_gate_blocked`。
- PG-327/328/329/330 wrapper 的入口分别强制 `CUDA_VISIBLE_DEVICES=0`；父循环还要求恰好一个可见设备、`cuda:0`、A800 名称和 `set_device(0)`。wrapper 不替调用方设置该环境变量，所以远程启动器必须显式导出 `CUDA_VISIBLE_DEVICES=0`。
- 只读证据已保存为 `research/pg331_remote_a800_readonly_preflight_v1.json`，文件 SHA-256=`b03fde1dbcf1d02fc9a9915de977d2438116dc38e811b502b2565634ad098937`，规则已登记该合同；最新 `research/improvement_rules.json` SHA-256=`d22840bef24063716ff120f7e736cd60ef0e5d9d376d062c5fc8d9d8a6ddf922`。远程资源空闲不解除 PG-331 source-row、信息保真和 fresh holdout 门。
- 代理审计确认首轮 collector 只采 `root GET` 与 `sqli_id POST`，evaluator 四项布尔值均故意为 false、split=`unassigned`、operator_reviewed=false，因此即使网络成功也只能得到 diagnostic/ASK，不能训练；历史 smoke 的 target-unavailable 风险仍需以新鲜 preflight/relay readiness 结果区分，不能把连接失败算成页面缺失或漏洞阴性。最小后续顺序是：relay/service readiness → 完整七轴 field manifest → evaluator-side candidate/reference/negative + typed evidence → failure/belief trajectory → source/implementation/family/entropy/capacity 审计 → 才能 A800 smoke。
- 研究台已接入两个只读证据 projection：`pg331_information_preservation.legacy_web_manifest` 与 `.remote_a800_readonly_preflight`，并新增 `pg331-legacy-manifest`/`pg331-a800-resource` blocked metrics；训练/promotion 逻辑未改变。`app/research_ops.py` SHA-256=`17c33f8a610dd4981e16fc3ac262417cf4ccae795944feb1ca616fa50cbf2c4f`，`tests/test_research_ops.py` SHA-256=`df60e0720377d3230bdbf85a83ae35b2205608a6ef4122dd3df0513ef35a1c2c`，研究台聚焦回归 `27 passed`。
- 展示层改动后的最终全量回归为 `1145 passed, 1 warning in 120.59s`；该结果仍只证明工程和证据展示合同，不提升 PG-331 训练资格。

### 2026-08-08 06:37 当前状态复核

- `execution_location_policy.training_schedule.weekend_remote_a800` 已重新解析确认：周六/周日、Asia/Shanghai、授权主机 `112.111.7.91:60228`、`NVIDIA A800 GPU0 only`、必须 `CUDA_VISIBLE_DEVICES=0`、GPU1–7 不触碰；周末资源许可不覆盖数据、信息保真、fresh holdout 或 promotion 硬门。规则文件当前 SHA-256=`d22840bef24063716ff120f7e736cd60ef0e5d9d376d062c5fc8d9d8a6ddf922`。
- 06:37 只读 preflight：`PG331_LOCAL_DOCKER_EVAL=1` 临时置于当前进程，固定镜像/资产/代码哈希/目标命名空间均通过，但因仍早于本地 `08:00–18:00` 窗口，`ready_for_diagnostic_collection=false`；`target_contacted=false`、`mutated_runtime=false`，四类 promotion 均为 false。preflight SHA-256=`58bb7716514eafcaa182f62559b1c87a7deaf8d3ae201152a4e2bb476977d7bf`。
- 远程只读报告仍为 `gpu_ready_data_gate_blocked`：GPU0 `NVIDIA A800-SXM4-80GB`、显存已用 `1 MiB`、利用率 `0%`、compute app=`0`；PG-331 信息保真和真实 source rows 门均未通过，training/payload/memory/vulnerability claim 全部关闭。
- 本轮只读回归 `tests/test_pg331_preflight.py tests/test_pg331_legacy_web_manifest_audit.py tests/test_research_ops.py`：`32 passed`。这只证明 preflight、历史清单诊断和研究台 projection 合同，不代表已有训练数据或漏洞能力。窗口开启后仍须先跑 fresh relay/GET+POST 采集；首轮预期是 `incomplete/ASK` 诊断，不得直接训练。
- 随后重跑信息/来源审计仍为 `blocked`：旧 PG-323 七轴 coverage 全为 `0`、context-target alignment=`0.082051`；PG-331 source-row collection `record_count=0`、`training_eligible_count=0`、failure=`missing:dataset`。本轮 source-row/loopback/trajectory 专项回归 `25 passed`，并未接触靶场或生成训练数据。
- 为修复“有 GET/POST 但没有参数化 GET”这一数据缺口，PG-331 首轮矩阵已扩为 3 条 neutral route：root GET、参数化 SQL GET（空 `name`/`submit`）和参数化 SQL POST（空 `id`/`submit`）。它们仍各自使用 fresh `network=none` 容器，route matrix 只在 evaluator 侧保存；模型上下文不含路由字面量、payload 或响应正文。collector SHA-256=`da0464f5673c8bdc8adb957aeda3d583c7658d0858f49f8dbea376ea55231baa`，collector 回归 SHA-256=`c1451660bcf4c8c433e8facff0622e5aae21c64f4bbeefb4b4c1c038d42dce21`；preflight 在 06:41 重新确认 `code_hash_lock_valid=true`，但仍因时间窗未到而不接触靶场。
- 适配器进一步修复参数信息熵：GET 查询键现在进入 `request_transport.parameters`，POST/GET 字段名只映射为 bounded semantic role（`identifier`、`query_term`、`submit_control`、`anti_csrf`、`destination` 等），不保留原始键；`param_role` token 因此可区分组合角色。适配器 SHA-256=`f41148fb21ce77d0e5ff49a2432fb1a30ef664a732c9a606b087dcf69a9e02a6`，tokenizer SHA-256=`31a1dd4f411e6d96d9314ac75b27a351a235828dc3488769521db459a663909a`；回归覆盖 `10 passed`（adapter + tokenizer）。词表仍是旧数据的 `diagnostic_only_audit_blocked`，只更新 tokenizer provenance，不把新角色伪装成已训练词表。
- 研究台已加入 PG-331 live collector projection：缺 artifact 显示 `pending`，单边 artifact 显示 `blocked_incomplete`；完整诊断时展示 route、GET/POST、parameterized GET、target contacted、ASK 与 training eligibility，并强制 training/promotion 全 false。当前 `app/research_ops.py` SHA-256=`bf88abc563a0ecbe7b2c8dc153f0db3f3ce4cd71afbabc53d4c31ad23280788f`、`tests/test_research_ops.py` SHA-256=`dfeb649da7e6ea2d2d2591d89fd73af36709f3b7dd101d61f36e88cf696e6503`；研究台回归 `29 passed`，未启动 Docker/GPU。

### 2026-08-08 06:51 工程回归与窗口前状态

- 研究台 projection 最终修复了参数化 GET 推导的 Python 生成器语法问题；`python -m py_compile app/research_ops.py` 通过，研究台 `29 passed`、PG-331 focused `3 passed`，全量回归为 `1148 passed, 1 warning in 123.85s`。当前 app/research_ops SHA-256=`5fddcd84f8c70ccaea38680d05183ad476d75d60404fe36c44db3bdb9b82cb38`。
- 06:51 preflight 再次确认 `code_hash_lock_valid=true`、固定镜像/资产/目标命名空间通过，但 Asia/Shanghai 时间仍为 `06:51`，所以 `ready_for_diagnostic_collection=false`、`target_contacted=false`、`mutated_runtime=false`；没有启动 Docker、训练或 A800 作业。
- 当前规则文件 SHA-256=`7abae710c9b67d7ace3676587b3ea83872fcc15579d835b7fc93b86fbbee0f91`。适配器 SHA-256=`f41148fb21ce77d0e5ff49a2432fb1a30ef664a732c9a606b087dcf69a9e02a6`、tokenizer SHA-256=`31a1dd4f411e6d96d9314ac75b27a351a235828dc3488769521db459a663909a`；词表仍 `diagnostic_only_audit_blocked`，没有把角色补丁或旧清单伪装成训练数据。
- 规则新增版本化 `pg331-parameter-role-v1` taxonomy：`identifier/query_term/submit_control/anti_csrf/destination/account_identifier/hidden_field/named_field`，原始参数名禁止进入 context，taxonomy append-only；当前规则文件 SHA-256=`6811af7bf05e0245c38aff0a90a05d171907141356655fbdd09d9fdb9765f0e2`。
- 词表构建器已读取该 taxonomy，将 8 个合法 `param_role=*` 值加入 ontology inventory；重建结果仍为 `diagnostic_only_audit_blocked`，但 inventory 从 `622` 增到 `630`、context vocabulary 从 `678` 增到 `686`，没有删除任何旧轴。代表整页容量仍为 required window=`497`，PG-322 legacy max=`72` 继续 FAIL，PG-331 minimum/balanced 仅容量 PASS，训练资格仍 false。builder SHA-256=`f2c905c8f73577e60f59645410dcd7b3b36b575054d71ededc3917346e0f772f`，vocabulary internal SHA=`091ecf139f4421dd16b004361a8ce1bef16a4b5cef29bf83416e0aba84542c5d`，capacity internal SHA=`2619cca2dd05c6c27985162e1e15650490f6daaef055c6545c9626f9698c6f53`。
- 本次生成后规则文件 SHA-256=`75d1642f4a766697a7a6cbd90793c5816a387123e62f1528b4f24ab5dc47fa76`；词表文件 SHA-256=`d96de22ffeed47eb2f772982bc7d3385ceb5efe8fd5fb657f23559eb4f0d9ffb`，容量报告文件 SHA-256=`fa208f6036cd0fbbf5409ab0f807eca93156dde4ed93686c31c2ed667534f086`。信息审计仍是 blocked，未提交 A800。
- 词表/容量 artifact 更新后的最终全量回归仍为 `1148 passed, 1 warning in 121.19s`；这只证明工程合同和信息保真实现，没有产生训练资格。06:55 preflight 仍因窗口未到而 `ready_for_diagnostic_collection=false`。

### 2026-08-08 07:02 周末 A800 规则结构化复核

- `research/improvement_rules.json` 的 `execution_location_policy.training_schedule.weekend_remote_a800` 已明确写入周六/周日（`Asia/Shanghai`）远程主机 `112.111.7.91:60228`，`gpu_index=0`、`NVIDIA A800 GPU0 only`、`CUDA_VISIBLE_DEVICES=0`，并保持 `other_gpus_touched=false`。
- 该许可只表示资源位置可用，不跳过 source-row、信息保真、fresh holdout、哈希锁和 promotion 硬门；当前 `research/pg331_source_row_collection_v1.json` 仍不存在，未启动训练或靶场。
- JSON 解析通过，资源规则回归 `tests/test_fast_local_docker_replay_policy.py -k weekend_remote_a800` 为 `1 passed`；规则文件 SHA-256=`2509c8d35c7dd81de8e2448fceb250db5161fd9ede3e9745f27e36b004e1ce4c`。
- 当前时间 `07:02` 仍早于本地 PG-331 采集窗口 `08:00–18:00`；窗口开启后的下一动作仍是 fresh Pikachu GET/参数化 GET/POST 诊断采集，不能直接提交 A800 训练。

### 2026-08-08 07:12 PG-331 evaluator-side typed sidecar

- 新增 `app/pg331_evaluator_sidecar.py`（SHA-256=`d2c53cde71c7fdbed70b987d0701b8b99144f64f9486da48f605b84af4a21697`）与 `tests/test_pg331_evaluator_sidecar.py`（SHA-256=`df4e88a804e90b960c012cdc1da710a53623c4a0e8106019a0c3176804a99bd2`）。它只接受脱敏 evaluator projection，不发请求、不启动容器、不训练。
- sidecar 固定 candidate/reference/negative 三角色，保存 fresh reset、typed effect、negative/reference/replay 检查及 record-role-bound evidence SHA-256；raw payload/response/oracle key 会 fail-closed，`model_context` 只保留抽象 availability/presence。
- `confirmed_positive` 仅代表 evaluator 侧本地 typed effect；`training_eligible`、payload/memory/vulnerability promotion 永远为 false，不能把 sidecar 结果当成通用漏洞能力。
- 专项 sidecar/preflight/资源规则回归当时为 `12 passed`，JSON 解析通过。合同已登记为 `pg331_evaluator_sidecar_contract`；随后 preflight 哈希锁更新，当前规则文件 SHA-256 见本节末。
- sidecar 尚未接入 live collector；当前时间 `07:12` 仍早于本地采集窗口，source rows 依旧为空，下一动作仍是窗口内 fresh GET/参数化 GET/POST 采集。
- sidecar 接入后的全量回归先为 `1155 passed`；加入 preflight sidecar-hash-lock 测试后的最新全量回归为 `1156 passed, 1 warning in 123.10s`。warning 仍只是既有 Torch nested-tensor 提示，不代表训练或能力晋级。
- preflight 现在也锁定 evaluator sidecar 文件/测试哈希：`scripts/preflight_pg331.py` SHA-256=`93ff20c3faec90feea761041fdeef9cdc9051ea7b364a7ae42708b8754301334`，并继续锁定 typed replay planner；07:27 实际 preflight 的 `code_hash_lock_valid=true`，但因时间窗未到仍不接触靶场。当前规则文件 SHA-256 见本节末。

### 2026-08-08 07:27 PG-331 typed replay planner

- 新增纯规划/绑定脚本 `scripts/run_pg331_typed_replay.py`（SHA-256=`5d685d6feba64b733a61ab97bf955ffebdf7ab26c094060949985a37dab7f648`）与测试（SHA-256=`c3cba784646dad31b2908fbf8a8f50bc1c2c5913ec7fcf52851d899ae8f747af`）。固定三条 Pikachu SQL row-shape GET/POST 路由、`network=none`、每 route/role fresh container 规划；不启动 Docker、不发包、不训练。
- candidate/reference/negative literal probe 只允许 evaluator 内存使用后做 digest，不进入 model context、trace 或 catalog；缺 typed/fresh/negative/replay 仍是 incomplete/ASK，promotion 全关闭。
- preflight 现在同时锁定 typed replay planner 与测试哈希；07:27 实际 `code_hash_lock_valid=true`，仅因时间窗未到而拒绝采集。PG-331 相关回归 `49 passed`；随后全量回归为 `1162 passed, 1 warning in 121.62s`。当前规则文件 SHA-256=`b00b7de5164d69cf267364b6188ce5e2ba38f582b9ad1e4aab6b9083f3bcc84f`。

### 2026-08-08 07:48 PG-331 decoder capacity/hash hardening

- 复核发现 PG-295 的实际实现是 decoder-only 单一词表、embedding 与 LM head 权重 tied；旧容量审计却把 context/target 当成两套参数。`scripts/audit_pg331_model_capacity.py` 已改为按 union(context,target)+`[PAD]`+`[UNK]` 计数，当前 `model_vocabulary_size=719`、`model_vocabulary_sha256=164bf9347730f476cce60d0597075237101f5bb0ffbec6e337b4bed9126c8505`，代表整页 required window 仍为 `497`，legacy `max_length=72` 仍 FAIL，PG-331 `768` 仅容量 PASS。审计报告仍因信息审计 blocked，`status=blocked`、`audit_sha256=99a66a6cace3e98f3a3df8e4ae00cad80fc20328a1aeae3e0a3425f9c404da90`，报告文件 SHA=`66228b3a967c1e9ae3840e7d4b16878d283ec14bd1a48d46409bbb5c4b29953b`。
- `scripts/run_pg331_a800_next_token_smoke.py` 现在在训练门内校验 capacity report 的内部哈希、model vocabulary size/sha，并拒绝 source row 中不在 append-only vocabulary 的 token；不再把未知字段静默映射到 UNK。PG-295 `_batch` 已按 `config.max_length`（而非旧硬编码 128）构造 batch，长页容量不会被静默截断。
- 训练合同与 preflight hash lock 已扩展到模型实现和容量审计：model=`app/pg295_causal_moe.py` SHA=`18ca917aced1a9a500b62d49b02abc1fc17a3e52288c5e0eecd107ba432880c0`；capacity audit=`scripts/audit_pg331_model_capacity.py` SHA=`c79de3df8af86ec86bd42bf839edef309f346c6dffe7e34a10ff23eb559f9579`；capacity test SHA=`76dc6f8bcad146ee05bf7a8828a41913128a7bb0ce13273517d9d3a1eebfca87`；training runner SHA=`fb16c15c6156222a0fb398d4b5fb4af6ff676cd92d1236643061e60b4fd07dca`；training gate test SHA=`51da7f6e84fcafc445ae18805f84bc46d5b08bef34d71b1a8fc39f363de178f8`；preflight SHA=`ee87d27bee3348498350d8e876806e178d8fa3b8b657df60efc66ae5111e23d4`。
- 07:45 只读 preflight：固定镜像/资产/代码 hash/目标命名空间均通过，但仍早于 `08:00–18:00` 本地采集窗口，`ready_for_diagnostic_collection=false`、`target_contacted=false`、`mutated_runtime=false`；训练脚本无显式 A800 环境时也 fail-closed。全量回归最新为 `1167 passed, 1 warning in 127s`（既有 Torch nested-tensor warning）。规则文件当前 SHA=`989223d38b5d8cb8d624cef08e00baf6e6d22d902d580ee96bda0fc80566481a`。
- 当前下一动作仍是 08:00 后先运行只读 preflight，再用 fresh pinned Pikachu 容器采集真实 root GET、参数化 GET、参数化 POST；首轮预期为 `incomplete/ASK`，不能直接训练。只有真实七轴 field manifest、typed evaluator candidate/reference/negative、fresh reset、证据哈希、轨迹/来源/实现/族外/熵审计全部通过，才允许周末远程 `112.111.7.91:60228` 的 A800 GPU0 短 smoke；promotion、payload catalog、vulnerability claim 继续关闭。

### 2026-08-08 08:27 PG-331 typed GET/POST source-row fresh replay

- 08:00 后首次 typed runner 尝试未产生能力结果：relay 的 Docker name-template 参数错误，且固定 Pikachu 的启动脚本需要 `/run/php` 与最小文件权限能力；该轮 `row_count=0`、`target_contacted=false`，仅保存为工程失败诊断，不得当作靶场阴性或训练数据。
- 修复 `app/pg331_pikachu_docker_relay.py`：`_exists` 使用合法 Docker 模板；移除会遮蔽 `/run/php` 的 tmpfs；保留 `network=none`、`no-new-privileges`、无 bind/volume、pids/memory 限额，并仅增补启动脚本所需的 `DAC_OVERRIDE/CHOWN/FOWNER/SETUID/SETGID` 五项 capability。relay SHA-256=`a9f69bea3e5761657ca625d3d0c78d155cf4179706710b941c8ff2f1abb7cfa1`，规则 hash lock 已同步。
- `app/pg331_loopback_adapter.py` 现在对无 Content-Type 但明显以 `<!doctype`/`<html`/`<head`/`<body` 开头的响应在内存中解析 HTML，避免把 doctype/lang 误标 unknown；不保存正文。adapter SHA-256=`6e9a29dd6bea25dbe63d9e7e23c3009420f7c869c77fca0ec2d8902fbb31b3bb`，新增缺 Content-Type 回归后 adapter tests=`14 passed`。
- 第二次 fresh typed replay 已完成：固定 Pikachu digest、`network=none`、每 route/role 独立 disposable container；3 路由（SQL string GET、SQL search GET、SQL numeric POST）× candidate/reference/negative/replay，实际 `GET=2`、`POST=1`，共 12 次容器回放。report `status=completed_diagnostic_only`，`row_count=9`，`typed_positive_routes=3/3`，candidate/reference/negative/replay、negative clean、fresh reset、evidence hash、replay consistency 全通过，`errors=0`，`target_contacted=true`；operator review 未开启，`training_eligible=0`。
- 抽象数据产物：`research/pg331_pikachu_typed_source_rows_report_v1.json` SHA=`cf7b295ccd378757332a43cbdd471180c704fb1fbbe9450719a389b90fa07d67`；dataset SHA=`61537a00efc8f935cab7b7a2138c83ff83ae5b9cb1315ca6f6452890e1503f98`；evaluator sidecars SHA=`6c7896d0771dda51ee55262b50a553a620f31cf8297605f0be928c5e2d53bdbc`。sidecar 只保留抽象 response-shape/effect 与 role-bound evidence SHA，不保留 raw probe/response。
- `scripts/audit_pg331_source_rows.py` 对这 9 行的结果为 `status=blocked` 但 `validation_counts.valid=9`、`context_firewall.forbidden_token_count=0`、`split_isolation.status=clean`；唯一阻塞是 `empty:training_eligible_rows`（operator review pending，且仍只有一个 implementation/family，尚未做跨来源/族外熵审计），不是漏洞阴性。audit SHA=`fff2dc0970e774796b952cb01bf1ac7a4b2ad778e0d6c9bdedb9f5976d1492d8`。
- 规则 `pg331_typed_source_row_contract.latest_run` 已登记上述 artifact/count/hash；所有 training/memory/payload/vulnerability promotion 仍为 false。下一安全动作是补第二个实现或第二个 SQL 表面并做多 seed/跨族 source holdout 与字段熵/消融；不要把这 9 行直接合并 canonical training dataset，也不要启动 A800 训练。

### 2026-08-08 08:39 PG-331 实际页面容量复核

- 对 fresh Pikachu typed source rows 直接测量发现每行 `context_tokens=3165–3284`（9 行均值约 `3226.89`），而非旧代表页的 365；保守 `required_context_window=4145`。这证明“768 足够”对真实页面不成立，不能训练前静默截断。
- `scripts/audit_pg331_model_capacity.py` 新增 `--dataset`、`--information-audit`、`--report` 参数，并保留默认旧审计兼容；自定义数据集的容量结果写入 `research/pg331_pikachu_typed_model_capacity_audit_v1.json`：status=`blocked`、information audit 仍 blocked、model vocabulary=`719`、required window=`4145`、最小候选/平衡候选均按 `max_length=4145` 才容量通过。报告文件 SHA=`3de5d6e48228e645800f951de04b93dba11f822fc2c23086faa59b14e73184ee`，内部 audit SHA=`ecb361843325ce9867fd839eb83a842ad5996dff21b6dc2f44a0082f7a534a4d`。
- `scripts/run_pg331_a800_next_token_smoke.py` 的训练门现在使用 `effective_max_length=max(768, measured_required_context_window)`，并把该窗口传给 CausalMoEConfig；容量不足时 fail-closed，绝不把长网页切成 768 token。训练 runner SHA=`48127826af99f28e93a9d410b8f9a083135ab8a65397b911c1ef381c86d7228b`，capacity audit SHA=`d67db70301facb715a890464acd8ff6a16c2b25175d97c6e2af54bdc6355390e`，capacity test SHA=`811836baefd891f9c36d29fd2f9947da44f1d54c73e81aab70d71c3aacb6fe9f`；相关专项 `11 passed`，preflight `code_hash_lock_valid=true`。
- 研究台已接入 typed source-row report/audit/sidecar 与容量 projection（`app/research_ops.py` SHA=`07e36bd2e626377c5e96521e5a7288b76ea188bc30006d4dbbef76859c396f56`，tests SHA=`3bda359b38fb7fe632e194f68b5e60cf1a1c95de5b0bd9102a71d4a556b49d08`）：显示 3 routes、9 rows、typed positive 3/3、GET=2、POST=1、operator review=false、training=0、audit=blocked、context max=3284、required window=4145，词表=1066，缺 artifact 时 pending；promotion 全 false。
- 结论：词表不是唯一容量门，真实页面长度也必须从 source rows 测量；在第二实现/族外数据、信息审计、字段消融与 operator review 前，不运行 A800。规则文件最新 SHA 在本轮结束后重新计算，不能使用旧快照中的哈希。

### 2026-08-08 08:47 PG-331 词表覆盖与字段熵审计

- 新增只读 `scripts/audit_pg331_dataset_information.py`：逐 7 轴、107 ontology fields 计算 presence/entropy/unique sequence、字段/轴消融、context firewall、typed/fresh/negative/replay、split/implementation 和容量覆盖；状态只允许 `blocked/incomplete/diagnostic`，`accepted_training_eligible_count` 恒为 0。当前 fresh 9 行审计为 `diagnostic`，唯一失败 `single_implementation_diagnostic`；有效行 `9/9`、context/target unknown token `0`、ontology inventory missing `0`。
- 审计报告 `research/pg331_pikachu_typed_dataset_information_audit_v1.json` 文件 SHA=`8c5209aa9536bdaea4990c54dd5c98f877e26ff9226939ec126093ad5743f20f`，内部 SHA=`843b1065e7d64358221b3bf5f7e2b65968f07cef6f979be18651b1ce70906f86`；脚本 SHA=`ba94c429e474f661022b4fec05cd3e496fcd6ffcc2dc0d2a2a0e2e95928e8f41`，测试 SHA=`9537c4e6f2319a4a185b79d19ea1763c7c1d4cee3118b921c8eed9d04b498896`，新审计 focused `6 passed`。
- `scripts/build_pg331_web_token_vocabulary.py` 支持基线词表 + 新 source rows 的 append-only 合并，补齐每字段 `observed/absent/not_observed/unknown` inventory；fresh typed 诊断词表 `research/pg331_pikachu_typed_web_token_vocabulary_v1.json`：context=`1054`、target=`33`、ontology inventory=`737`，文件 SHA=`34539b8c2034a9170c908eee486c7d901f3edc752a40d0b22551918663c33656`，status 仍 `diagnostic_only_audit_blocked`，不训练。builder SHA=`415a247eca579bc263e1f5d30f0a546297f50a3629f7e8c67dc937c8e2b4cb15`，测试 SHA=`879273f32b44024da9aa206725fc22159731c9e8f0c8fab739101856dae35fdc`。
- 用 append-only 词表重算 fresh 容量：model vocabulary=`1066`、context=`3165–3284`、required window=`4145`、inventory missing=`0`；`research/pg331_pikachu_typed_model_capacity_audit_v1.json` 文件 SHA=`8012c5ff3c07fc31d0bede1e91102e9a3ac4e824adc4e6110c7f466cfe17f428`，内部 SHA=`bc1ca5b755bc8bf9b535912cb5e0b99b76aa3f4cbd3598dae9d10612daeeceaa`，status 因信息审计非 passed 仍 blocked。capacity audit 当前 SHA=`b412356bec84bd0375d86e61fb1be7861a9efb5186ec35bf3790d07dc9ffa80a`。
- 结论：旧 686/54 词表会把真实网页大量 token 当未知，已用 append-only 诊断 manifest 修正覆盖；但只有一个 Pikachu implementation，不能把 `diagnostic` 误升为训练资格。下一动作是第二独立实现/族外 source rows，再重复同一审计，确保信息熵与容量门在留出集上仍成立。

### 2026-08-08 08:58 PG-324 第二实现缺口静态审查

- 对现有 Juice Shop PG-324 trace/catalog 做了只读字段审查，没有启动 Docker/GPU/网络。旧 trace 只有 12 个 coarse context key；可复用 typed/fresh/replay/evidence/belief 摘要，但缺 document 18 字段、navigation 12 字段、request 19 中大部分字段、response 12 中多项、JavaScript 16 字段和 failure 13 字段，且 candidate/reference/negative 没有统一 role-level fresh reset identity。
- 因此不能把 PG-324 trace 转成完整 PG-331 gold/source-row；最多作为 evaluator-side partial diagnostic，缺失字段必须显式 `not_observed/ASK`，training/memory/payload/vulnerability 全 false。PG-324 trace/catalog 仍是 evaluation-only，不能靠历史 route/canary 补齐信息熵。
- 新增审计后的全量回归：`1187 passed, 1 warning`。当前无运行中的训练、采集进程或 PG-331 容器。规则文件最新 SHA=`c08a839aeaaed093fe6387477fedf99fb0139452a6262fabb09aec5c935358cc`。
- 下一轮最小可行实验：为第二独立实现重新做 fresh per-role candidate/reference/negative/replay，采集完整 7 轴/107-field manifest 和同一 4145-window capacity audit；只有两实现的字段熵、消融、族外最坏 seed、ASK/repair/negative 门都通过，才考虑远程周末 A800 GPU0 smoke。

### 2026-08-08 08:59 周末 A800 GPU0 权限复核

- 规则无需重复新增：`research/improvement_rules.json:50-62` 已明确 `execution_location_policy.training_schedule.weekend_remote_a800`，仅周六/周日（Asia/Shanghai）允许授权主机 `112.111.7.91:60228` 的 `NVIDIA A800 GPU0`，`gpu_index=0`、`CUDA_VISIBLE_DEVICES=0`、GPU1–7 不触碰；仍要求操作者消息、数据/代码/审计哈希锁和信息保真门，fresh holdout 前 promotion 保持关闭。
- 08:59 只读 SSH 再核对 GPU0：`NVIDIA A800-SXM4-80GB`、81920 MiB、已用 1 MiB、利用率 0%，未发现 GPU0 compute app；未查询或触碰 GPU1–7，也未启动训练、容器或远程写入。资源可用不等于训练门通过。
- 当前 PG-331 仍是单一 Pikachu implementation 的 `diagnostic`/`training_eligible=0`，真实页面要求 `required_context_window=4145`；下一动作仍是第二独立实现 fresh source-row 采集与字段/熵/族外审计，不能因为周末许可直接提交 A800 smoke。规则文件当前 SHA-256=`c08a839aeaaed093fe6387477fedf99fb0139452a6262fabb09aec5c935358cc`。

### 2026-08-08 09:43 周末 A800 GPU0 规则确认

- 再次验证 `research/improvement_rules.json` 可解析，`execution_location_policy.training_schedule.weekend_remote_a800` 明确为 `Saturday/Sunday`、`Asia/Shanghai`、`112.111.7.91:60228`、`gpu_index=0`、`CUDA_VISIBLE_DEVICES=0`；`other_gpus_touched=false`。
- 资源许可不等于训练许可：仍需显式操作者消息、数据/代码/审计哈希锁、信息保真门和 fresh holdout；promotion、长期记忆、payload catalog、vulnerability claim 继续关闭。此次未启动训练、容器或远程写入。
- 规则回归 `tests/test_fast_local_docker_replay_policy.py -k weekend_remote_a800`：`1 passed`；当前规则文件 SHA-256=`d1b3c7b73c10ce3dd647d4bf3938db2d6f00eb9de974cb0afbde37d6811866f4`。

### 2026-08-08 11:xx PG-331 v4 诊断与存储治理

- 适配器发现并修复了一个真实的信息标记问题：已完整捕获的请求中没有 CSRF-shaped 参数，或响应头明确没有 Content-Type/charset 时，分别记录为 `absent`；只有未捕获/传输失败才保留 `unknown`。未知不会被默认值覆盖。`app/pg331_loopback_adapter.py` SHA-256=`631755343e9026bcf29baf8d1ed9fa88990851f3dfefd2541badbdfbbddf5af4`，`tests/test_pg331_loopback_adapter.py` SHA-256=`96f6ad3801d311167c91fed3c1caeaf8377a3ba609b09b50a0144c73a65a37e1`；该轮 adapter/source-row/live focused tests=`29 passed`（测试文件完整 SHA 以规则锁为准）。
- Juice Shop v4 使用固定 image digest、`network=none`、每 role fresh disposable container 和浏览器 DOM/JS 抽象投影；由于 POST 路由 cleanup 的 Docker 查询返回 `CalledProcessError`，只完成 2 GET 路由：6 行、typed positive=1、GET=2、POST=0、errors=1。该产物明确是 `incomplete`，不是失败漏洞阴性，也不进入训练。v4 report/dataset/sidecar/source-audit/vocab/info-audit/capacity 文件及 SHA 已登记在 `research/improvement_rules.json:pg331_juice_shop_source_row_live_contract.latest_run`；目标容器已按精确名称清理。
- v4 6 行的 source-row validation `valid=6`、context firewall=0、七轴 presence observed、字段 inventory missing=0；但信息审计仍因单一 implementation 和数据集实际 required window=`4603`（容量审计 blocked）而不能训练。`training_eligible=0`、memory/payload/vulnerability promotion 全部 false。该结果只证明采集器能保留页面 token 与已知缺失状态，不证明模型能力。
- 存储盘点：C 剩余约 43.0 GiB，D 剩余约 41.1 GiB，E 剩余约 200.2 GiB，F 剩余约 89.8 GiB；D 工作区约 47.7 GiB，其中 `artifacts` 约 46.0 GiB。将 22 个历史候选权重目录（每目录至少 0.5 GiB，合计约 34.0 GiB；不含当前 PG-331/最近冻结 evidence）移动到 `E:\blackboxanalyze-archive\artifacts\legacy-large-20260808`，并在 D 原路径建立 junction，未删除内容、旧脚本路径仍可复现。归档清单为 `research/storage_archive_manifest_v1.json`，清单 SHA-256=`adf715b86239310bb7ff69a866ffe91939f585b1f82255a508e7c4759682ffe6`；归档后 D 余量约 78.0 GiB、E 余量约 169.6 GiB。C 盘 Codex/环境缓存暂不移动，避免破坏运行时；后续只处理有明确缓存语义的目录。
- 本次 v4/存储/POST 诊断更新后的 `research/improvement_rules.json`（含 storage_governance）SHA-256=`0e6d53a078bdc815ad16d1cb705cefd54e9d09b5fb26a2cfd2c5ff804b8db061`；周末 A800 仅允许远程 GPU0（`112.111.7.91:60228`, `CUDA_VISIBLE_DEVICES=0`），但 source-row、信息熵、容量、跨实现/族外 holdout 未过前不得训练。
- 本轮最终全量回归为 `1207 passed, 1 warning`；warning 是既有 Torch nested-tensor 提示。该结果只证明工程合同、哈希锁和存储变更没有回归，不改变 `training_eligible=0` 或 promotion 全关闭。
- 补充 POST-only fresh 诊断（seed=33112）完成 1 路由/3 role 行：`typed_available=false`、typed positive=0、negative clean=true、replay consistent=true、`question=ask_typed` 全部 3/3、`safe_to_send=false` 全部 3/3、errors=0。它证明 POST typed 不可用时会安全 ASK；由于 evaluator typed 缺失，严格 source-row audit 仍 blocked，绝不进入训练。report/dataset/sidecar/audit 与 SHA 已登记为 live contract 的 `supplemental_post_run`。

### 2026-08-08 12:21 PG-331 seed33113 多种子诊断与审计器修复

- 在 `08:00–18:00 Asia/Shanghai` 本地诊断窗口内，以固定 Juice Shop digest、`PG331_LOCAL_DOCKER_EVAL=1`、`network=none`、无挂载、每 role/replay fresh disposable container 完成 seed=`33113`。3 条 allowlisted route（GET track-order、GET products-search、POST login unsupported）共 9 source rows + 12 个 role/replay 生命周期，目标均已清理。
- 结果 report=`completed_diagnostic_only`、`errors=0`、`target_contacted=true`、GET=2、POST=1；track-order GET candidate/reference/replay typed effect=`true`、negative clean；products-search GET typed effect=`false`、negative clean；POST typed evaluator=`unavailable`，因此 3 行统一 `ASK/ safe_to_send=false`。`typed_positive_routes=1/3`，`training_eligible=0`，promotion/memory/payload/vulnerability 全 false。该 positive 只表示本地 evaluator 的可复放状态，不是通用漏洞能力或可迁移 payload。
- 产物：report 文件 SHA=`91964fe5b38fc7455be36049853a5023c5eab1c05e7938cbb4daea9f40b215b2`；rows=`5254f4952ede071cf8844d08fd2d40f0d2f5a59b8c975188dfdb5d2f767ede82`；sidecars=`a965095eef5a5583c43e57b98ac9764f5957b8d30861aad947712fcc5b1867ac`。source-row audit 文件 SHA=`a16929bc1eccb088eb1ce3684d29490616ce434070aa74839d5d4b4ff9268dc`：9 行中 6 valid，3 个 POST 行因 typed evaluator 缺失被严格拦截；status=`blocked`。
- append-only 词表构建完成：context=`1193`、target=`41`、ontology inventory missing=`0`；vocab 文件 SHA=`0c0e4c731130011928362b2daaa1ba57ab0c0fdaaca6a9cb8f1081148e1cf4e9`。信息审计 status=`incomplete`：unique sequence ratio=`1.0`、七轴字段均有 manifest，但只有一个 implementation，且 typed/replay 不完整；context firewall forbidden token=`0`。dataset required window=`4610`（context max=`3656`），PG-322 legacy 72 继续 FAIL，PG-331 minimum/balanced 仅容量 PASS；capacity audit status=`blocked`，文件 SHA=`cbbc17b5b898d67b050d4ddbd07ec8374c2bf215d9a1b1bc161c2ad94f380c46`。
- 真实工程缺陷：`scripts/audit_pg331_source_rows.py` 对 CLI 相对路径直接调用 `Path.relative_to(ROOT)` 会抛异常；已增加 `_dataset_label()`，同时支持 workspace-relative 与外部 staging absolute path，并新增 2 个回归测试。script SHA=`66c69b4f731d2ea611125567e084f801b9bff07e9daa7e40fe5f37752e2374e3`，tests SHA=`26fd3145e4be3c2ff2245e3f3c6468e18fe68bf604a46f75012181c4b51fbcff`，focused=`5 passed`；规则哈希锁已更新，preflight `code_hash_lock_valid=true`，规则文件 SHA=`2c23e3cd9db96b84703a37ee640ad35f1712ad03f0d24adc4941f2e2eabf80b7`。
- 下一动作仍是第二独立实现/族外 source rows 和多 seed 交叉审计；在 uniform field/typed/fresh/negative/replay/entropy/holdout 通过前不提交远程 A800，当前没有新的训练集或长期记忆。

### 2026-08-08 12:29 PG-331 跨实现诊断合并

- 新增只读诊断合并器 `scripts/merge_pg331_diagnostic_datasets.py`，要求至少两个输入，逐文件保存 SHA-256/报告内 dataset hash/record-id hash，拒绝重复 `record_id`，不改 split、context 或 training eligibility；测试 `tests/test_pg331_diagnostic_merge.py`=`3 passed`。script SHA=`8b88890d07e9932c1c670ff998ceba1ebba881dbd2490ecbb36525118cff31c5`，test SHA=`ccd050af5c68b072e43add302848a9fea958e42b27d1d0684c47cf8c50907481`。
- 将 Pikachu SQL 9 行与 Juice Shop seed33113 9 行合并为 `research/pg331_cross_impl_diagnostic_v1.json`（文件 SHA=`a90caef410fe1c3f7ce9781f49aa6b055c86baffccdad85028cb8926eebf69f7`）：18 行、2 implementation、3 family、unique sequence ratio=`0.944444`；source audit valid=`15/18`，3 个 Juice POST 行因 typed evaluator unavailable 仍 invalid/ASK。
- 合并信息审计：status=`incomplete`、implementation/source=`2/2`、fresh reset=`18/18`、negative=`18/18`、typed/replay complete=`15/18`、context firewall forbidden=`0`。轴熵（bits）为 document=`1.0`、navigation=`1.459148`、request=`2.251629`、response=`1.0`、javascript=`1.0`、failure=`1.459148`、belief/replay=`1.459148`；相比单一实现，信息多样性实际增加，但 `capacity_dataset_window`=`4610`、typed/replay 缺失和 invalid rows 仍让审计 blocked。
- 合并词表 `research/pg331_cross_impl_diagnostic_vocab_v1.json`（SHA=`b84cc2c7b06af97fc8b2d98c78685e965cf0898cae5e1b9f732cc1d744cf3435`）inventory missing=`0`；capacity audit `research/pg331_cross_impl_diagnostic_capacity_audit_v1.json`（SHA=`68739799f21af23acab071371b920ba7258b317ecd3a1d6b2aa0b7dedb340197`）status=`blocked`。合并数据仍不进训练/长期记忆/payload catalog，规则已登记为 `pg331_cross_impl_diagnostic_merge_contract`。
- 跨实现合并后全量回归：`1212 passed, 1 warning in 126.18s`；唯一 warning 仍是既有 Torch nested-tensor 提示。preflight `code_hash_lock_valid=true`、`no_existing_pg331_targets=true`；本轮没有训练、GPU 或残留 PG-331 容器。当前规则文件 SHA=`11d8b0e7a912623305e068c8a90a78e69652865c97c7328dfa9c07c01de25ab0`，AGENTS 快照 SHA=`3f3d43437d0c22487ee39043347cd942c91116930e2e5e1f32bde4cdfc853752`。

### 2026-08-08 12:55 PG-331 seed33114 与跨 seed 合并

- Juice Shop seed=`33114` 完成同一 3 route/9 row/12 fresh role-replay 诊断，report=`completed_diagnostic_only`、errors=`0`、GET=`2`、POST=`1`、typed positive=`1/3`；track-order GET typed candidate/reference/replay 正常、products-search GET 为阴性、POST typed unavailable→ASK，training/promotion 全关闭。report SHA=`ead3cb9a917b6536f99c00e9157050d1ead1aa9d5f3f5d0e07837ee9ad38ad99`，rows SHA=`0853aade0697c5d754c99c1e69197f9f4eb58bfc472ab05d5cfa503f05434815`，sidecars SHA=`b3fbe912ce7275dd3e1bd2b311fb242cf307fd23000ceb45485c5867e1ca5852`。
- seed33114 source audit valid=`6/9`（POST 三行 strict invalid），单 seed 信息审计仍 `incomplete`；容量 required window=`4610`，词表 inventory missing=`0`。完整审计文件已登记为 live contract 的 `seed33114_run`。
- 将 Pikachu 9 行 + Juice seed33113/33114 各 9 行合并：`research/pg331_cross_impl_multiseed_diagnostic_v1.json` SHA=`b9cda1fdc350e13a0ab7762df8c5d9aff8be8c9c097d22c4fe00fc3e54b14408`；27 rows、2 implementation、3 family、valid=`21/27`、fresh=`27/27`、negative=`27/27`、typed/replay complete=`21/27`，unique sequence ratio=`0.777778`。轴熵 bits：document=`1.351644`、navigation=`1.224394`、request=`2.281036`、response=`0.918296`、javascript=`0.918296`、failure=`1.530493`、belief/replay=`1.530493`。这说明跨 seed 增加了真实信息变化，同时也暴露模板重复；审计仍 `blocked`（typed/replay/容量/invalid rows）。
- 跨 seed 审计、词表、容量与 promotion 全部登记在 `pg331_cross_impl_multiseed_diagnostic_contract`，仍不可训练/晋级；规则文件当前 SHA=`8780676acebee0215aba290aa040174c7440f1370bcdc71061bc737b6cd0662`。本轮无训练、无 GPU、无残留 PG-331 容器。

### 2026-08-08 13:xx PG-331 train/holdout 硬规划门

- 新增 `scripts/plan_pg331_train_holdout.py` 与 `tests/test_pg331_train_holdout_plan.py`。规划器只读取 source rows，要求显式且不同的 train/holdout implementation；只接受已由 source-row contract 标记 `training_eligible=true` 的完整行；不重标 split、不输出 context token，只输出 row-id/digest/hash。focused tests=`3 passed`；script SHA=`003a25952d48ea7680bed55011ad4831af3887081905a6906fac3a548e91b4f5`，test SHA=`a5756f3711c3b55e5bf98e98b094ca2e692efed13f9e1269e419f8e5f40f393a`。
- 当前跨实现/跨 seed 合并集执行规划：`input_rows=27`、`validated_rows=21`、holdout=`18`，但 `eligible_train_rows=0`，且 Pikachu 行本身仍是 `implementation_holdout`；计划文件 `research/pg331_cross_impl_multiseed_train_holdout_plan_v1.json` SHA=`bc62b266996ffc39a167751e8386332a015a0915abc5a7acb67605bb621e2e74`，status=`blocked`，training/memory/payload/vulnerability 全 false。这个结果禁止通过手改 split 或把诊断行伪装成 gold 来启动 A800。
- 现有第三方 OWASP VulnerableApp 固定 digest（PG-246）虽有 GET typed DOM 与 POST 405 abstain，但旧数据只有粗粒度过程 token、缺 PG-331 七轴/107-field manifest；不能直接转换成训练行。下一工程动作是为该独立实现做严格 PG-331 whole-page adapter，或补一个同样有 typed POST evaluator 的授权实现；在此之前只保留 evaluator-only 证据。

### 2026-08-08 13:04 周末 A800 GPU0 规则确认

- `research/improvement_rules.json` 已明确：仅 `Saturday/Sunday`（`Asia/Shanghai`）允许在授权远程主机 `112.111.7.91:60228` 提交 A800 训练，且必须显式 `CUDA_VISIBLE_DEVICES=0`、`gpu_index=0`；GPU1–7 不查询、不触碰。
- 该条目仍要求操作者消息、数据/代码/审计哈希锁和信息保真门；fresh holdout、uniform contract、训练资格未通过前，`training/memory/payload_catalog/vulnerability promotion` 全部保持 `false`。本次只核对规则，没有启动训练、容器或远程写入。
- 规则文件当前 SHA-256=`39B5669AB7A340F727EDFD322E9C3B829A8D188973002502729D141DFD7EA6C1`。远端 GPU0 只读预检仍为 A800-SXM4-80GB、1 MiB 已用、0% 利用率、无 compute app；资源可用不等于训练门通过。

### 2026-08-08 13:xx PG-331 train split 诊断与跨实现留出

- `scripts/run_pg331_pikachu_typed_source_rows.py` 增加显式 `--seed`、`--split`、`--report`、`--dataset`、`--evaluator`，并限制输出路径在工作区内；默认仍为 `implementation_holdout`，不会静默改 split。脚本 SHA=`c9cb1f3f25618bc5dc8b2a908fe933741fcfa6577e770ab16b229a4803232bf0`，既有 typed-source tests SHA=`67c267058c69cb663927aab2d40223380d9422faed84a2e531208f51ed9b46ea`，py_compile 与专项测试通过。
- 在本地诊断窗口以固定 Pikachu digest、`network=none`、每 role/replay fresh disposable container 运行 seed=`33115`、split=`train`：3 routes（GET=2、POST=1）、9 rows、typed positive=`3/3`、errors=`0`、operator_reviewed=`false`、training_eligible=`0`；目标已全部清理。报告文件 SHA=`d59a90b9ae920b06dce63ff2071f6190d41e43ec297a9463ba49bc341b7e85de`，dataset 文件 SHA=`8f691cb192ccebe998dbcd927eb29488f02c2d91c1c7d1c74b28f29b27f051a4`，sidecars 文件 SHA=`94e4225aef507d46064189b5129a0342d5ef01fde95d1cc5a9bf9691cca8f007`。这是真实本地 GET/POST 诊断证据，不是可训练 gold。
- 将 seed33115 Pikachu train rows 与 Juice Shop seed33113/33114 holdout rows 合并为 `research/pg331_train_holdout_diagnostic_v2.json`：27 rows、train=9、holdout=18、2 implementations、3 families、source valid=`21/27`、invalid=`6/27`（Juice POST typed unavailable）、fresh=`27/27`、negative=`27/27`、replay complete=`21/27`；source audit、append-only vocab、information/capacity audit 和规划器均已生成。合并文件 SHA=`44faae799232c2badfdba2256328dd18f14425fd44caee4913896b64c2697d3e`，规划 status=`blocked`，eligible train rows=`0`。
- 关键容量结果：context max=`3656`，保守 required window=`4610`，模型词表=`1257`；legacy max_length=72 仍失败，PG-331 4610-window 变体仅表示容量可容纳，不能越过信息/复放/审核门。promotion、A800 training、长期记忆、payload catalog、vulnerability claim 全部保持关闭。
- 存储审计没有发现新的安全可删对象：C/D/E/F 可用约 `83.43/152.70/166.15/89.83 GiB`；历史大权重 33.995 GiB 已在 E 归档并由 D junction 保持复现，当前 PG-331/研究证据、源码和 `.codex` 均未移动或删除。
- 本轮没有启动 A800；下一安全动作是完成独立实现的严格七轴 collector/evaluator 与显式审核，再重新跑信息熵、容量和 train/holdout 规划。
- 本节追加后规则文件 SHA-256=`d8a635dae0ece3d18fad0c89b877df754dbb2ac1b0c453c9da9f90599b20ef25`；上一节的 `39B566...` 是规则变更前快照，仅作历史证据。

### 2026-08-08 13:24 回归与 canonical audit 恢复

- 新 train/holdout 诊断期间曾误用默认 CLI 覆盖 `research/pg331_source_row_audit_v1.json`；已立即用缺失 canonical dataset 的只读审计恢复，当前 `record_count=0`、`training_eligible_count=0`、failure=`missing:dataset`，文件 SHA=`d4b4ea3d661f0d988a12b792cb817acc91253804f1fef1ab74c8f3d2119d0f6d`，内部 audit SHA=`e746410e44c0d722554228fc37c3a9b7262cc8342e92e17c03262ffa30bc8b2b`。seed33115 审计仍在独立 v2 文件，不影响 canonical 研究台契约。
- 恢复后全量回归通过：`1218 passed, 1 warning in 125.28s`；唯一 warning 是既有 Torch nested-tensor 提示。PG-331 专项仍为 `104 passed`，preflight `code_hash_lock_valid=true`、`no_existing_pg331_targets=true`。
- 当前规则 SHA=`9a33fadd20e0490dfa64ca720beb7b1dc6324205363b38fe84b682b5679e1aa0`；AGENTS.md 的当前哈希请以阶段结束时 `Get-FileHash` 结果为准，避免在文件内自引用自身哈希。

### 2026-08-08 13:xx VulnerableApp 结构适配器硬门

- 新增纯内存 `app/pg331_vulnerableapp_adapter.py` 与 `tests/test_pg331_vulnerableapp_adapter.py`：复用 `_PageParser/_field_capture_manifest`，输入仅是已脱敏 markup/headers/request/response projection，输出七轴 observation 与 107-field manifest；302 结构可保留，缺观测显式 `not_observed`，原始 URL/payload/response/body 字段拒绝，测试覆盖 GET/POST/302/缺失/旁路泄漏。
- 重要修复：适配器没有 evaluator 输入，因此即使 GET 有响应也统一 `typed_available=false`、`next_action=ask_typed`、`safe_to_send=false`；typed effect 必须由独立 candidate/reference/negative/replay sidecar 绑定，不能把页面结构当漏洞证据。adapter SHA=`186d33c823ab41dd1293cfa1557e67b30924d2afb0dd957830038b24cae049b3`，tests SHA=`dd65eac4de395936ed9936f489ac20306fca3682395581b4d9bd400a20e64d2c`，focused=`28 passed`。
- 该适配器仍未连接 Docker/网络/训练；下一步是把它绑定到独立 VulnerableApp fresh lifecycle 和 evaluator-side 角色回放，完整 source-row/information audit 通过前不得生成训练 gold。

### 2026-08-08 13:35 适配器修复后的周末 A800 决策

- 适配器修复后的专项回归为 `28 passed`，全量回归为 `1224 passed, 1 warning`（唯一 warning 仍是既有 Torch nested-tensor 提示）；`py_compile app/pg331_vulnerableapp_adapter.py` 通过。此次只验证代码合同，没有启动 Docker、靶场、训练或远程写入。
- 周末远程资源规则保持不变：`Saturday/Sunday`、`Asia/Shanghai`、`112.111.7.91:60228`、`NVIDIA A800 GPU0`、`CUDA_VISIBLE_DEVICES=0`；GPU1–7 不触碰。当前只读预检显示 GPU0 空闲，但资源可用不是训练资格。
- PG-331 train/holdout v2 仍为 `27` 行、`eligible_train_rows=0`、`typed/replay` 不完整、信息/容量审计 blocked；因此本轮不提交 A800 smoke，不生成训练 gold、不提升长期记忆、payload catalog 或 vulnerability claim。下一安全动作仍是独立实现的 fresh 七轴 collector + typed candidate/reference/negative/replay sidecar，再重新做字段熵、容量和族外留出审计。
- 当前规则文件 SHA-256=`9a33fadd20e0490dfa64ca720beb7b1dc6324205363b38fe84b682b5679e1aa0`；AGENTS.md 不在自身内容中嵌入自引用哈希。

### 2026-08-08 13:48 PG-331 reviewed train candidate 与槽位对齐修复

- 以用户已授权的本地最终复核开关 `PG331_OPERATOR_REVIEWED=1`，在固定 `sift/pikachu-fixed@sha256:cca428...f472c6`、`network=none`、每 role/replay fresh disposable container 下完成 seed=`33116`：3 routes（GET=2、POST=1）、9 rows、typed positive=`3/3`、errors=`0`，其中 candidate/reference 6 行 `training_eligible=true`，negative 3 行保持非训练；容器已清理。
- 独立 source-row audit 首轮暴露 `context_target_alignment=0`，定位为审计器把 Rule-IR 的抽象 `parameter_role/encoding_chain` 槽位误当成必须出现的短文本，而真实上下文使用 `request_transport_field_parameter_role/request_transport_field_encoding_chain` 前缀字段。已修复 `scripts/audit_pg331_source_rows.py` 的显式槽位别名映射；`probe_variant_ref` 保持 evaluator-bound、继续不要求把 evaluator literal 放入模型上下文。缺少对应上下文字段仍会阻断。新增审计测试后 `tests/test_pg331_source_row_audit.py=7 passed`，seed33116 source audit=`passed`、alignment=`6/6`、context firewall=`0`。
- seed33116 词表 inventory=`0 missing`、context=`1092`、target=`59`；信息审计为 `diagnostic`（唯一硬缺口=`single_implementation_diagnostic`），容量 audit=`blocked` 但 required window=`4145`，PG-322 legacy 72 仍失败；promotion/training/memory/payload/vulnerability 全部 false。该批只作为 reviewed single-implementation candidate，不能直接上 A800。
- 研究台新增只读 `pg331_train_holdout_diagnostic_v2` projection（app/research_ops.py SHA=`4d4a897aca01cfdb9841782c971b0b805d69c225671eb3dda0a6d10e736b7380`，tests SHA=`d8514bba726e86be1fd1bab1fc0d5a85faf357c3f5e99c35c2de95bb0c1119c3`）；focused `2 passed`、全 research_ops `35 passed`。projection 只显示摘要/哈希/熵/容量，不输出 rows/tokens/payload/response/oracle。
- 证据文件：report SHA=`93bd3dbb8c61273fb7fd700821b1f8c15491d843ae39bf4870b2307bbd86fec3`、dataset SHA=`58530e48777f477ad967ee237542f1db8af1ac60c172f28da8c545012f89cd0b`、sidecars SHA=`00f5cc30c17f9f53414ba9774efd63fc45f2a1737ec817c1c75aaea0fd5fdd6`、source audit v2 SHA=`3fbe9161d501597acbdf3961753e9e1d03d33feb77f4d640468115affa970420`。
- 当前 `research/improvement_rules.json` SHA-256=`e76002aa519071c143203638064d707ed670cb1c288baae1dcf10ed45d447208`；preflight `code_hash_lock_valid=true`。下一步仍是第二独立实现的完整 typed GET/POST source-row collector，随后重跑跨实现信息/容量/holdout 审计；本轮未启动 A800。

### 2026-08-08 13:50 PG-246 live collector 网络契约阻断

- 对 PG-246 旧 VulnerableApp 生命周期做了静态复用审查：其专用 bridge host-only network 不满足 PG-331 的 `network=none + loopback relay` 硬门，不能把旧 bridge 结果重新标记为严格 source row。新建 planning-only `scripts/run_pg331_vulnerableapp_source_rows_live.py` 与测试，固定 3 seeds × 6 cases × candidate/reference/negative/replay，GET/POST 能力显式记录，缺 typed POST 必须 ASK，`run()` 在网络契约不满足时 fail-closed；未启动 Docker、网络或训练。
- live collector focused suite=`12 passed`（连同 adapter/plan），文件 SHA：runner=`79c80ca6662f8a33d72a6258b5f4d8109743edf1bfcdafd2f11786d9ec84bdb9`、tests=`b1203de225098ef4b67875bf8a1542e82f641ad5fc998a8c6bacb120aea424b0`、contract=`c0cd549791d67fb89716fd419606701e1a77a3d8a8510941584b83c8412ef97c`。该规划产物不是 source data，promotion/training 全 false。
- 当前规则文件 SHA-256=`7b19c365504943cc389d20823f122e8271cb88ff48d31f6d73cfca3fd2b96b90`；下一步是单独设计并人工复核 network-none relay lifecycle 与 typed role-bound evaluator，不能静默复用 bridge。PG-331 reviewed Pikachu 候选仍不得上 A800。

### 2026-08-08 13:55 全量回归确认

- 本轮修改后全量 `python -m pytest -q --durations=10` 通过：`1231 passed, 1 warning`（唯一 warning 为既有 Torch nested-tensor 提示，耗时约 133.6 秒）。PG-331 专项/研究台 focused 也已通过；没有启动 A800、Docker 或外网目标。
- 当前可训练候选仍只是一种实现的 6 行，独立实现 live collector 仍是 planning-only；因此“测试全绿”只证明工程合同，不改变 `training_allowed=false`、promotion 全关闭。
- 关键当前哈希：`audit_pg331_source_rows.py`=`df55454e571c7f32001e508c5c89a70e180f436d70de6a7b01597ef5a00defb7`，`app/research_ops.py`=`4d4a897aca01cfdb9841782c971b0b805d69c225671eb3dda0a6d10e736b7380`，规则=`7b19c365504943cc389d20823f122e8271cb88ff48d31f6d73cfca3fd2b96b90`。AGENTS.md 自身哈希不写回自身。

### 2026-08-08 14:xx reviewed train/holdout 规划与 relay helper

- 将 seed33116 reviewed Pikachu 9 行与 Juice Shop seed33113/33114 各 9 行合并为 `research/pg331_train_holdout_reviewed_seed33116_v1.json`：27 行、2 implementations（Pikachu train / Juice holdout）、3 families，6 eligible train、18 holdout、21 valid。`scripts/plan_pg331_train_holdout.py` 规划 `status=passed`，但它只证明 split/实现隔离，不授权训练；计划文件 SHA=`98e85138df57bc441abde874148a8d560d4d010f7d564915905b3a301edaa8c7`，内部 plan SHA=`631b5c855781bf663486941b50dde8fd7382aa38cb59ff2f301bb2815c3fa4c7`，合并文件 SHA=`e2e23336c475cb9870f6c0a9ebace18d8c71b8808271b9dc0d821c208e0c7198`。
- 完整 source-row audit 明确拦截 6 个 Juice POST typed-unavailable 行（其余 21 valid）；不把缺 evaluator 的 POST 当负例。合并信息审计 `incomplete`（`invalid_rows/typed_evaluator_incomplete/replay_state_incomplete`），词表 context=`1213`/target=`65`、inventory missing=`0`，容量 required window=`4610`、legacy72 FAIL，故仍不能训练。
- 新增纯 `app/pg331_network_none_relay.py` 与测试，固定无 publish、无 bind/volume、`network=none`、127.0.0.1 relay、目标只允许 `http://127.0.0.1:9090/VulnerableApp`，response 仅状态/类型/长度/重定向投影；旧 bridge 明确 incompatible。focused relay/live/adapter tests=`12 passed`；helper SHA=`96d91239ccbb4e9c19d29997c3a8634d6c0f546b455680c970e98f4698535c20`，tests SHA=`c50f317e8e74394b774dd441d9f91694b94de3760cb13bd5d8302905be86c6ec`。仍需单独审核 relay 进程与 live gate，未启动 Docker/网络。
- 当前规则文件 SHA-256=`365097c7889c4217f309ea130ec95b8e3e5fd27ebe1e869a14ed060b644664bf`；A800、长期记忆、payload catalog、vulnerability claim 继续关闭。

### 2026-08-08 14:06 PG-331 有效行漏斗与信息审计

- 新增 `scripts/select_pg331_audited_rows.py` 的显式 `--materialize-valid-output` 路径；默认 manifest 仍只保存不可逆行引用，物化文件只复制逐行校验通过的抽象 source rows，保持原 split，不重标、不复制 raw payload/response/oracle/evaluator literal，并强制每行及外层 `training_eligible=false`。当前脚本 SHA=`1676b8e494b9b2f43b02219b9f8b53e33c7954b45c3ceecfd3ff4504ed0e4104`，测试 SHA=`cc31f0a7c4b87a2c3ef2ac67bbf907e24c8f3514332c1d4c2d1166cd3d5c199e`，focused=`14 passed`。
- 对 `research/pg331_train_holdout_reviewed_seed33116_v1.json` 执行漏斗：输入 `27` 行，逐行有效 `21`，显式排除 `6`；train/holdout 原样保留为 `9/12`，两种 implementation。六条 Juice Shop POST 因 `typed_available` 缺失以及 belief/replay 字段未完成，保持 `ASK/incomplete`，没有重标成阴性或训练样本。manifest SHA=`91feccf43f48cb3416a173f9ad55cf203b4a94a7c1b1ae84bdc182700941dd1a`，物化诊断 rows SHA=`a2902793a9dafd168bcc7138c638409f9eebbc1a3e322e4ceff8e83839a38354`。
- 对 21 条有效抽象行的重审结果：source-row strict audit 仍只作诊断（物化时清除训练资格）；七轴/107-field 信息审计 `status=diagnostic`、无硬失败、context firewall=0，信息文件 SHA=`cf168ba64223e9ab4ea04e7da6fe1b89cd18759f3ffd79b09f30b95eb3e6339f`；append-only vocab inventory missing=`0`、context=`1201`、target=`59`；容量审计 required window=`4603`、context max=`3650`、legacy72 FAIL、PG-331 minimum/balanced 容量可容纳但 promotion 仍关闭，容量文件 SHA=`828bf9f34ea98a54ee341bdc00c05e49c9b106babaaf85a691e8c5943ed54290`。
- 物化集的 train/holdout planner 明确 `blocked`（training_eligible input=`0`），因此它只能用于可复现的信息/容量诊断，不能被当作训练集。规则已登记为 `pg331_audited_row_funnel_contract`；当前 `research/improvement_rules.json` SHA=`14b99bab69e41c01d6f83c3102d0fe67529e9c6f01351bb4b01fe8144e1ab55d`。
- 本轮未启动 Docker、靶场、网络、A800 或训练；下一安全动作仍是为独立实现补齐 typed GET/POST evaluator、fresh role reset 和 belief/replay sidecar，再重新做 source/implementation/family/entropy/capacity 审计。周末 A800 GPU0 规则保持有效，但不绕过这些门。

### 2026-08-08 14:09 全量回归

- 物化漏斗与规则登记后的全量 `python -m pytest -q --durations=10`：`1237 passed, 1 warning in 130.54s`；唯一 warning 仍是既有 Torch nested-tensor 提示。该回归只证明工程合同，不改变训练资格。
- 当前 selector/materialized 资产与规则哈希已复核；规则文件仍为 `14b99bab69e41c01d6f83c3102d0fe67529e9c6f01351bb4b01fe8144e1ab55d`。没有启动 Docker、靶场、网络、远程 A800 或训练。

### 2026-08-08 14:xx network-none relay 身份绑定加固

- 代理完成第二实现 relay/collector 的静态合同加固：`build_container_command` 与 `build_relay_contract` 现在只接受固定容器名或严格的 `pg331-vapp-nn-<seed>-<case_sha_prefix>-<role>` 名称；每个 seed/case/role 的 relay contract 与 container identity 一一绑定，避免把同一容器误当 fresh。collector 同时要求每 role 的 reset evidence hash、network-none inspection、127.0.0.1 relay、无 published port/无 mount，replay 只在 evaluator 侧，POST typed 不可用时强制 ASK。
- 当前代码/测试 SHA：relay `app/pg331_network_none_relay.py`=`11d1d8c1334e0db723824125c92cd9067c76bafa46239800015feb7e3bece855`，collector `scripts/run_pg331_vulnerableapp_source_rows_live.py`=`94e2cd1a82bcaf3902d7c3024ec10375fc613be79d5198576dc0cbde0a695aab`，对应测试=`111b0b37b2def5c722b5f2141885f086ce3da171eca3fb6079fdf2a6c922ceab` / `1e4b8529af8f02bd99fdbc4be9a403e386bd36d0465f6dde8baa8b37cf88a8fa`；专项 `16 passed`。
- 该轮仍是 planning/attestation-only：没有启动 Docker、网络、GPU 或训练。真正 live relay 进程、逐 role Docker inspect/reset、GET typed candidate/reference/negative/replay 仍需单独审核；旧 PG-246 bridge 不能重分类。规则文件已更新并解析通过，当前 SHA=`764383b065ede9da1a531c7f129b1cb1f7d783e1ec48b1197d86d7c3c8d8bae4`。

### 2026-08-08 14:18 PG-332 候选盘点与回归

- 只读盘点了本地 WebGoat、DVWA、Pikachu variants：PG-146/PG-207 只有 login surface/unknown oracle，旧 PG-72 虽有 typed GET/POST 但是 coarse trace，不能升格为 PG-331 source rows。`sift/pikachu-pg240-source-native@sha256:de3227c1f56969be94521bc4bb48814b5dd1f511a1e368c688933812eaafe973` 有旧 typed GET/POST，但与 fixed Pikachu 共享 runtime，只能作为 source-heldout/variant，不能未经独立性证明当第二实现。WebGoat/DVWA 若接入 PG-332，必须新增完整七轴 adapter、candidate/reference/negative/replay typed sidecar 和 fresh reset，不得复用旧粗粒度数据。
- relay/collector 加固后的全量回归：`1238 passed, 1 warning in 130.41s`；唯一 warning 仍为 Torch nested-tensor。preflight code hash lock 通过，当前没有 Docker、网络、GPU 或训练运行。
- 当前下一工程优先级：先为真正独立的 WebGoat 或 DVWA 设计 planning-only PG-332 adapter/evaluator contract；在第二实现完整 typed GET/POST、字段审计、跨实现留出之前，不上 A800、不训练、不提升长期记忆或 payload catalog。规则已登记 `pg332_independent_lab_selection_contract`，最新规则 SHA=`19e37c668010bffab932c2f134a7b59afb35ce40070540e89eff63692145deaa`。

### 2026-08-08 14:22 PG-332 DVWA planning contract

- 新增纯 planning-only `scripts/plan_pg332_dvwa_source_rows.py` 与 `tests/test_pg332_dvwa_source_rows_plan.py`。固定 DVWA digest，内部仅保留 3 条 allowlisted lane 的 route hash（2 GET/1 个无状态 POST ASK lane），不输出路径、payload、响应、oracle literal；3 seeds × candidate/reference/negative/replay fresh identity，`network=none`、loopback relay、无 publish/mount，7 轴/107 字段全部 `not_observed`，模型目标固定 `ask_typed/safe_to_send=false`，promotion 全关闭。已移除 stored-XSS/任何持久化写入路线。测试=`2 passed`；script SHA=`13d96ee6bbc9899f8fbfb5b8dc0f6680faf93bb903e08d27b23d7e4e2540dbdc`，tests SHA=`c3c2c2981ab7085b5cfadb4c5b37d495535867fe9cc51cfd278414edde21afd3`。
- PG-146/PG-207 的 WebGoat/DVWA login 与旧 coarse trace 明确排除，不能转成 PG-331 source rows。DVWA 的 live relay/fresh lifecycle、typed GET candidate/reference/negative/replay evaluator 尚未实现，POST 保持 ASK-only；本轮未启动 Docker、网络、GPU 或训练。
- 规则已更新为 `pg332_independent_lab_selection_contract`，当前规则 SHA=`ad413abc7f0820bdcffc24c4bc36aaebd53e0471ec86f7908d154033ec18da09`。

### 2026-08-08 14:26 最终回归

- PG-332 规划合同加入后的全量 `python -m pytest -q --durations=10`：`1240 passed, 1 warning in 132.84s`；唯一 warning 仍为 Torch nested-tensor。preflight code hash lock 继续通过。
- 当前工作区没有运行中的 PG-331/PG-332 容器、训练进程或 A800 作业；所有新代码仍为 planning/attestation-only，训练、长期记忆、payload catalog 和漏洞声明保持关闭。

### 2026-08-08 14:32 大框架优先、按需细化与快速训练规则

- 用户确认研究流程采用“大框架先行、细节按需补齐”：先建立 document/navigation/request/response/JavaScript/failure/belief-replay 七轴骨架和 Rule-IR 槽位，再仅在 `unknown_field_needed_for_next_action`、typed oracle 缺口、失败修复、信息增益、阴性歧义或跨实现分歧出现时追加细节。没有观察到的字段继续标记 `not_observed/unknown` 并生成 ASK，不能静默填 `absent/negative`，也不能因追求速度删除低频轴。
- 训练前必须完成一次可复核的 planning bundle：操作意图、数据/代码/审计哈希、schema 与 context firewall、信息熵/容量摘要、train/holdout split、fresh target 与 typed evaluator 状态、资源设备门。bundle 未锁定时不启动训练；锁定后允许批量执行，避免每个微步重复全量审计。
- 快速路径只减少重复检查，不降低硬门：训练期间监控 next-token loss、ASK/abstain、failure action-change、negative false-accept、holdout predictive entropy、灾难性遗忘 canary 和 GPU0 资源；checkpoint/early-abort 必须保留。任何授权范围、raw-context firewall、fresh reset、正/参考/阴性 typed evidence、证据哈希、信息保真或 operator review 门失败，立即 quarantine，不通过降阈值或删检查换速度。
- PG-332 的 `dvwa-xss-stored-post` 现被纳入用户授权的 evaluator-only stateful disposable lane：允许 synthetic stored-XSS/数据库挑战状态，但每个 seed/route/role 必须独立 fresh container、reset 前后、database-clean attestation、teardown/restart、`network=none` + 127.0.0.1 relay、无 publish/bind/volume/external network；状态差分、原始状态、payload、响应只留 evaluator-side。缺 typed candidate/reference/negative/replay 或任一 reset/clean/evidence 字段时仍为 ASK/incomplete，training/memory/payload/vulnerability promotion 全部 false。PG-324/PG-325 等既有 read-only lane 的 no-stateful-write 规则不变。
- 本轮规则/计划哈希：`research/improvement_rules.json`=`0d90b4abd0486398b005fc4735b2dfd3af78ba75b829ae7c1cff0029101f4f5b`；`scripts/plan_pg332_dvwa_source_rows.py`=`9e4fd76106dff3eee3e1fe2092ca6e6bb14b8767df254cf91ed8f8f1d4441c85`；`tests/test_pg332_dvwa_source_rows_plan.py`=`80709e38ab00ab21982ec3638f0ced326541c679b0f2b3ba5139a38d4392f436`；`research/pg332_dvwa_source_row_plan_v1.json`=`4aecbd67dddfdfa917328bce96b2ec136d106ea1cd2a8cdc9bdbdd2d45d614cd`，内部 plan SHA=`41016c0d1ba8b816d802be020fc55681a3712540d48975ac57b91ea74f3648e4`。本轮只改规则/备忘录，未启动 Docker、网络、训练或 A800；下一动作仍是实现并审阅 PG-332 live relay/evaluator，然后再做跨实现信息审计。
- 规则/计划更新后的专项与全量回归均通过：PG-331/PG-332 relay/adapter/plan focused=`16 passed`；全量 `python -m pytest -q --durations=10`=`1241 passed, 1 warning in 131.50s`。warning 仍是既有 Torch nested-tensor 提示；测试没有启动 Docker、网络、训练或 A800，也没有改变当前 `training_allowed=false`。

### 2026-08-08 14:xx A800 表示层候选与存储归档

- 为满足“A800 不空闲”而不绕过 capability 硬门，新增 context-only `scripts/run_pg331_a800_representation_smoke.py`。它只读取 train split 的抽象 `context_tokens`，holdout 只在训练后用于 context-only 评估；不读取/复制 `target_tokens`、payload、响应、oracle 或 evaluator authority。固定词表外 token、缺 field manifest、row firewall 非零、非 GPU0 A800、未锁数据/代码/词表/规则哈希都会在 CUDA 训练前阻断。所有结果标记 `representation_pretrain_candidate_only`，training/memory/payload/vulnerability promotion 永远 false。
- 第一轮发现真实架构问题：tied token/LM-head embedding 使用 `nn.Embedding` 单位方差默认初始化，1213 词表下初始 logits 可达约 120，训练前 predictive entropy 已接近 0，loss 约 91。已修复 `app/pg295_causal_moe.py`：新增 `initializer_range=0.02`，显式初始化 token/position embedding；当前模型 SHA=`f7d9b94d0a6859770c5ff89813d4008716664e4cb3a7294008e68a7a774dccec`，PG-295 与 representation 专项测试通过。
- 修复后在授权远程 `112.111.7.91:60228`、`CUDA_VISIBLE_DEVICES=0`、A800 GPU0 做了 3 seed × 4 epoch context-only smoke：train=9 行、implementation holdout=18 行、required context window=3656、holdout vocabulary unknown=0；train loss=`6.864788/6.779052/6.815169`，holdout loss=`6.924949/6.890615/6.924382`，平均熵约 `7.075`，不再塌缩。报告=`research/pg331_a800_representation_initfix_lr1e4_e4_v1.json` SHA=`f2525e184820c27131309a38114906da86d56190fb85d7053e14fe7764e3df8b`，checkpoint SHA=`5488260bcfbebf22bd02491bb32e410b7853c48ce50b12d3501e37480672f052`。这是表示层候选稳定性证据，不是 capability/SFT/RL 成功；信息审计仍 incomplete、长期记忆和 payload catalog 关闭。
- representation runner 当前 SHA=`f140eb979afb949051e8082215fb7abcfe3a5250e3eddc4b4631c8cb89e48961`，focused tests=`8 passed`。旧初始化候选 v1/v3 已标记 superseded，不得作为 teacher 或长期记忆。
- 存储治理：按只读审计将 25 个明确历史 PG147–PG216 权重目录（10,688,629,389 B，约 9.955 GiB）从 D 归档至 `E:/blackboxanalyze-archive/artifacts/legacy-pre-pg217-20260808`，D 原路径保留 25 个 directory junction；未删除、可逆恢复。清单=`research/storage_archive_manifest_v2.json` SHA=`252b705f684afc0bf8bccc16b8568e042cf2806832718e4c2214e03aa66385d2`，规则当前 SHA=`5c4f6c66ed661c74d74a7c605e1a5e7118ee3d556bbf89f5fab3c7fa62d51142`。归档后 C/D/E/F 可用约 `83.23/161.15/156.19/89.83 GiB`；PG-331/332、research、app、scripts、tests 未移动。
- 下一动作：补第二独立实现的完整七轴/typed GET+POST source rows；表示层可继续 A800 candidate smoke，但 capability SFT/RL 仍必须等待跨实现信息/容量/negative/fresh/replay 硬门。全量回归需在本节后追加最新结果。
- 本轮架构初始化、representation runner、存储规则和归档后全量回归：`python -m pytest -q --durations=10`=`1249 passed, 1 warning in 131.46s`；warning 仍为既有 Torch nested-tensor 提示。当前没有运行中的 Docker、靶场或训练进程，远端 A800 smoke 已结束并清理到只保留候选 artifact。

### 2026-08-08 15:xx PG-332 fail-closed runtime contract 与 A800 8 epoch 表示对照

- PG-332 DVWA collector 继续保持“可审阅但默认不接触目标”的 fail-closed 运行合同：`scripts/run_pg332_dvwa_source_rows_live.py` SHA=`71531eb59a0b6edf2958196dde33c97a15e2605716cf22c1050a5925196ee6b8`，测试 SHA=`d1444ee8408a088fa354dc1fc3dd09a2c8319b9a6fc44887ea8b9f6a3d226614`，专项=`7 passed`。即使显式设置 `PG332_LOCAL_DOCKER_EVAL=1`，当前仍只返回 `incomplete_environment_failure/ask_typed/safe_to_send=false`，因为 DVWA login、network-none relay 和 typed evaluator 尚未接线；未启动 Docker/网络，也没有把旧 bridge 重新分类。所有 promotion 仍关闭。
- 远程 `112.111.7.91:60228` A800 GPU0 在周末空闲预检通过后，追加 3 seed × 8 epoch 的 context-only 表示候选 smoke，`CUDA_VISIBLE_DEVICES=0`，未读 `target_tokens`、payload、响应或 evaluator authority；运行结束后 GPU0 恢复 0%/无 compute app。报告文件=`research/pg331_a800_representation_initfix_lr1e4_e8_v1.json`，文件 SHA=`fbf02051f27641dc8a323a5b1930bc38ed743525d195bd02508e4306529fb6c`，内部报告 SHA=`cb57a5d07554479dc54e4351021276667be8410093c8951127de5de62caef30d`；checkpoint=`research/pg331_a800_representation_initfix_lr1e4_e8_v1.pt`，SHA=`1a8e4332ef470f4c8adee5d737718a02f03c5e0285e3444c734cf4428eabeacc`。
- 8 epoch 结果：train loss=`6.689260/6.619538/6.636167`，implementation holdout loss=`6.789055/6.763608/6.801087`，逐 token 平均 predictive entropy=`7.073285–7.074526`，未再出现预测熵塌缩；状态仍为 `representation_pretrain_candidate_only`，information gate=`incomplete`，training/memory/payload/vulnerability promotion 全部 false。该结果只证明初始化修复后的表示层稳定性，不证明 Rule-IR、SFT/RL 或漏洞检测能力。
- 本轮规则已把 PG-332 fail-closed runtime SHA 与 8 epoch 报告登记；`research/improvement_rules.json` 当前 SHA=`724d11c8764c5ba77f8be327cd5ca1b898a999f77759b6ca964c69593ae47c24`。下一项仍是接通并审阅 DVWA 的真实 network-none relay/typed evaluator，再重跑七轴字段、跨实现/族外和容量审计；没有这些证据不得启动 capability SFT/RL。

### 2026-08-08 15:xx 回归与资源核验

- PG-332/PG-331/模型专项回归=`45 passed`；随后全量 `python -m pytest -q --durations=10`=`1253 passed, 1 warning in 130.44s`。唯一 warning 仍是既有 Torch nested-tensor 提示；测试未启动 Docker、外网或新的训练作业。
- 本机当前无运行中的 PG-331/PG-332 容器、训练进程或 docker-exec relay。远端 A800 GPU0 的 8 epoch 作业已结束，post-check 为 1 MiB、0% utilization、无 compute app，GPU1–7 未查询/未触碰。
- 存储复核：C/D/E/F 可用空间约 `83.12/160.93/156.19/89.83 GiB`；25 个历史目录仍在 `E:/blackboxanalyze-archive/artifacts/legacy-pre-pg217-20260808`，D: 原路径 junction 保留，manifest=`research/storage_archive_manifest_v2.json`（SHA=`252b705f684afc0bf8bccc16b8568e042cf2806832718e4c2214e03aa66385d2`），删除标志为 false，可逆恢复。PG-331/332、当前 research/app/scripts/tests 未移动。

### 2026-08-08 末：PG-332 relay 台账同步（短收尾）

- 本轮没有再启动 A800、Docker 或训练。远端 A800 8 epoch context-only 表示候选已经完成；当前只做哈希/进程收尾，避免在信息门未过时继续消耗时间。
- PG-332 代码台账已同步到当前实现：runner SHA=`4a8fa5ef03073f0ef977701a2e504635ec7e304591f4eb363b21c6f6beecf346`，live tests SHA=`057b8187073bf0a7d6f5e51510015e0a006eb7556c1c614af77abe3c2054210f`，network-none Docker relay SHA=`08654e3675273f12b5a16325691a2ab2b609ebc6171dfc35c06c41b5c88a059f`，relay tests SHA=`6eb865fc0af081fbc8fee5b154686a94edab2b28d13921761974b0f3a8ecd8c6`；focused live+relay=`6 passed`，JSON/py_compile 均通过。
- relay 仅允许固定 DVWA digest、`network=none`、无 publish/bind/volume、127.0.0.1 内存桥接与 disposable reset；typed candidate/reference/negative/replay evaluator 尚未接线，故仍为 `incomplete_environment_failure/ASK`，training/memory/payload/vulnerability promotion 全部 false。旧 bridge 不得重分类。规则文件当前 SHA=`fe75cab4fb851870c54508ac033c7cfbac678062f4bf4cc921f9fe8538c5f243`。
- 当前无 `pg332` 容器、docker-exec relay 或训练进程。不要把这次工程测试的全绿当作漏洞能力，也不要把表示层候选当作 SFT/RL teacher。

### 2026-08-08 16:xx PG-332 DVWA typed GET 跨实现诊断与快速表示候选

- DVWA 已在固定 digest 的 disposable `network=none` 容器中完成数据库初始化/认证健康门；新增 evaluator-only inert HTML shape typed GET 回放（candidate/reference/negative/replay），3 seeds、9 rows、candidate/reference typed=`3/3`、negative violation=`0`、replay=`3/3`、context firewall=`0`。这只证明受控响应形状差分，不是脚本执行或通用 XSS 结论；原始 probe/response 仍只在 evaluator 内存侧。
- 新行：`research/pg332_dvwa_typed_get_source_rows_v1.json`（SHA=`9ea3a8692d1c36f67a4de82e37447b191ccf9616c0adfa006fdb93fe14d9c1e0`），source audit=`d26503d24a71fbbd63a4ba4fcaf014a4f0ae3bdc4cef4d21d754f268a040845d`。与 Pikachu seed33116 9 行合并为 `research/pg332_dvwa_pikachu_cross_impl_source_rows_v1.json`（SHA=`67b303176e398bfbd1858d6cffc4d7ba1475f418d76b116e177359c46148a446`）：18 行、train/implementation_holdout=`9/9`、2 implementations、7 轴/107 fields 全覆盖、typed/fresh/negative/replay=`18/18`、unique sequence ratio=`0.666667`、firewall=`0`。信息审计=`diagnostic`（SHA=`079baf8e2316bf88875c4fd87f15b7148d7fa70d6d864d6ac91830f1b7cf5e6d`），容量 required window=`4145`；source/信息审计不自动授权 capability 训练，accepted rows=`0`、promotion 全关闭。
- 周末远程 A800 GPU0 已用 context-only 表示候选快速 smoke：主机 `112.111.7.91:60228`，`CUDA_VISIBLE_DEVICES=0`，3 seeds × 1 epoch，train=9、implementation holdout=9，未读取 target/payload/response/oracle。报告=`research/pg332_dvwa_pikachu_cross_impl_a800_representation_smoke_v1.json`（SHA=`9ae640943cfc635e85b0a181c18061e4abba58000930a7c75dbb63781b60b604`），checkpoint SHA=`c9c762ee44c4f154bbd75bc2db9365db292437c82560f41278176e1b37ce47ce`；train loss=`7.043461/7.089596/7.034742`，holdout loss=`7.055483/7.075152/7.029621`，平均熵约=`7.0187–7.0205`，无词表外 holdout token。状态=`representation_pretrain_candidate_only`，信息晋级门=false，GPU0 结束后恢复 0%/无 compute app。
- 当前规则文件 SHA=`bf9ff3aa929b2411fe4893276040571b972f6912c592306fcce3ce9961143135`。新增 `research_execution_policy_v1`：大框架优先、按需细化；可跳过重复/装饰性检查但不跳过授权、network-none、fresh reset、GET/POST 真值、三角色 typed 证据、context firewall、split 和容量门；持久状态 stored-XSS/数据库 lane 允许在每角色 fresh/clean/teardown 的 evaluator-only 范围内测试，缺证据仍 ASK，training/memory/payload/vulnerability promotion=false。
- 下一步：接入并审阅 stateful stored-XSS POST evaluator，完成 GET/POST pair 后重跑 source/implementation/family/entropy/capacity 审计；在此之前不做 capability SFT/RL，也不宣称模型能检测任意网址漏洞。

- 末尾回归：`python -m pytest -q --durations=10`=`1257 passed, 1 warning in 135.00s`；规则 JSON 解析通过，PG-332/PG-331 focused=`35 passed`。本次回归未接触 Docker/外网，A800 smoke 已结束；规则最终 SHA=`bf9ff3aa929b2411fe4893276040571b972f6912c592306fcce3ce9961143135`。

### 2026-08-08 16:xx PG-332 stateful stored-XSS POST 与 GET/POST 跨实现矩阵

- 新增真实 evaluator-only `scripts/run_pg332_dvwa_typed_stored_post_source_rows.py`：固定 DVWA digest、`network=none`、loopback docker-exec、每 seed×role/replay 独立容器；先 baseline，再 POST，再复读页面，只把抽象 state delta/shape/evidence hash 交给 source-row，原始 marker/表单/响应/数据库状态全部留在 evaluator 内存。三 seed × 4 role=`12`，candidate/reference typed=`3/3`，negative violation=`0`，replay=`3/3`，fresh/database-clean/teardown 全通过，9 rows/6 claimed eligible；报告=`research/pg332_dvwa_typed_stored_post_report_v1.json` SHA=`f543ea98b981ce558cdfaab7677f71738d45ece3e199d87a5ad893c6099d3e63`，rows SHA=`c6ea95c9d35d34c1e8e33814550ced60c7afd086e5733b8fe1c10a4d10799bf5`，source audit SHA=`90932145e301ecd30837e9b5a3edea57b818a4d996a07d1f46138962a3794b98`。仍不是通用 XSS/任意网址漏洞声明。
- sidecar 现在显式区分 `database_touched=true + disposable_state_delta=true + state_delta_class=disposable_evaluator_state` 与外部业务写入；只有固定 evaluator 状态可被硬门接受，训练/记忆/payload/vulnerability promotion 仍 false。对应 sidecar SHA=`9ae94a683dd09f9c52131f56019a32c4f7298c7cb273516cd421e46c89670d37`，focused stateful/sidecar=`19 passed`。
- GET+POST 合并诊断集：`research/pg332_dvwa_pikachu_get_post_cross_impl_source_rows_v1.json`，27 rows（Pikachu train=9、DVWA implementation_holdout=18）、2 implementations、valid=27、typed/fresh/negative/replay=27/27、unique sequence ratio=`0.555556`、context firewall=`0`、required window=`4145`；source audit SHA=`74d74d529775527813948c5d47de68eead41d3296c946f3fcdc921826d5c1fc8`，information audit=`diagnostic` SHA=`fbe577494ad139745c53d00f64c22a65849425c53c1ad2108130e8df0bfd0655`，capacity technical minimum/balanced=`true/true`，accepted training rows=`0`。
- 这批新上下文在周末远程 A800 GPU0 做了 3 seed × 1 epoch context-only smoke：报告=`research/pg332_dvwa_pikachu_get_post_cross_impl_a800_representation_smoke_v1.json` SHA=`0dee98dc6a977291ce92a3fe9274290bb4eb5ddc9a1a39b106fe134ecae5fa51`，checkpoint SHA=`747e68304e66932259c8438d5df88a8d394904aced2320537f73e32c3a5be3d6`；train loss=`7.066085/7.024827/6.999260`，holdout loss=`7.068669/7.043904/7.003199`，平均熵=`7.044434–7.047382`，无词表外 token，GPU0 结束后恢复 0%/无 compute app。仍为 representation candidate，不是 capability SFT/RL。
- 存储治理：把明确历史目录 `artifacts/pg147-model-capacity-sweep-v1` 与 `artifacts/pg163-large-typed-mix-v1` 移至 `E:/blackboxanalyze-archive/artifacts/legacy-pg147-pg163-20260808`，D: 原路径保留 junction；共 `624,803,392` bytes，未删除。清单=`research/storage_archive_manifest_v3.json` SHA=`7784e66ee6b6a31062d39873714656eff62397f7a54e5d74d5519f04f9d969e1`。C/D/E/F 当前约 `82.95/160.81/155.61/89.83 GiB` 可用；PG-331/332 当前证据未移动。
- 当前规则 SHA=`fc6165e4985913a73a1c9bf125a020f967ea9f5d73fc380bb940fbcf6a4ae98c`。下一步是把 stateful POST 与现有 GET/SQL rows 接入研究台 projection，再做完整回归；在信息/跨族/独立实现门通过前不启动 capability SFT/RL、不提升长期记忆。
- 最终工程回归：`python -m pytest -q --durations=10`=`1260 passed, 1 warning in 131.07s`；唯一 warning 仍为既有 Torch nested-tensor 提示。新 stateful sidecar/POST runner、旧 PG-331/332 合同均通过。

### 2026-08-08 末：PG-332 研究台证据投影与 A800 状态收口

- `app/research_ops.py` 新增 `_pg332_extended_diagnostic_projection`，把 DVWA typed GET、DVWA disposable stateful POST、Pikachu/DVWA 跨实现 source audit、七轴信息审计、容量审计和远程 A800 context-only smoke 统一投影到 `capability.model.pg332_diagnostic` 及 `pg331_information_preservation.pg332_extended`。投影只保留计数、熵/唯一序列摘要、容量、硬门布尔值和证据哈希；不输出 `records/context_tokens/target_tokens`、原始请求、payload、响应体、oracle/evaluator literal。所有 training/memory/payload/vulnerability flags 强制为 false。
- 当前研究台可见：GET=1 路、持久状态 POST=1 路；两 lane 各 3/3 typed positive seeds、negative violation=0、fresh/replay=3/3；合并 `27` rows、`2` implementations、typed/fresh/negative/replay=`27/27`、unique sequence ratio=`0.555556`、context firewall=`0`；信息审计=`diagnostic`、accepted training rows=`0`；required context window=`4145`；A800 context-only `3` seeds、implementation holdout=`18` rows、target tokens 未读取，information gate 仍未通过。
- 本次投影与测试 SHA：`app/research_ops.py`=`6f93f2d7cb16b1b0ebec67f9408daf05261170d6dd6288eb8add24eb98a6be3a`，`tests/test_research_ops.py`=`e469419597ac9868a8981fcd8d70f0b4efe3d47e2fcaaff42376a3c9ede259a19`；专项 PG-331/332=`11 passed`，全量 `python -m pytest -q --durations=10`=`1262 passed, 1 warning in 131.54s`。规则文件未因投影放宽，SHA 仍=`fc6165e4985913a73a1c9bf125a020f967ea9f5d73fc380bb940fbcf6a4ae98c`。
- 结论：A800 不是“没训练”，而是已完成一轮表示层 context-only 候选 smoke；完整 Rule-IR/SFT/RL 能力训练仍因信息保真、第三实现/族外留出和 accepted source rows 门关闭。下一安全动作是第三实现或族外 holdout 与失败/ASK 轨道，不是重复低信息训练。

### 2026-08-08 末：A800 四 epoch 表示层对照

- 在周末授权远程 `112.111.7.91:60228` 的 A800 GPU0 上追加 3 seed × 4 epoch context-only representation smoke，显式 `CUDA_VISIBLE_DEVICES=0`、`BLACKBOX_REMOTE_A800_TRAIN=1`；训练只读抽象 `context_tokens`，不读取/复制 `target_tokens`、payload、响应或 evaluator authority。GPU0 运行后复核为 1 MiB/0%/无 compute app，GPU1–7 未触碰。
- 报告=`research/pg332_dvwa_pikachu_get_post_cross_impl_a800_representation_e4_v1.json`（文件 SHA=`7096838b5bd79998e5523dada8d980d3e59039da4c899b0cd204571416170022`，内部 report SHA=`4df6360ad5e47f20d4d4b2f28ed562160e0621986e6d843a51b4c9010793b890`），checkpoint=`research/pg332_dvwa_pikachu_get_post_cross_impl_a800_representation_e4_v1.pt`（SHA=`5e1cd6eb72c4db673d2714ddc15b49b5d9d45ca33e2557a4ea36bd7e91e6ebf6`）。train loss=`6.846246/6.832714/6.807863`，implementation holdout loss=`6.886826/6.884118/6.850089`，holdout predictive entropy=`7.044446/7.045928/7.046322`，未出现信息熵塌缩；状态仍为 `representation_pretrain_candidate_only`、information gate=`diagnostic`、promotion 全关闭。
- 研究台已优先读取该 e4 报告；`app/research_ops.py` 当前 SHA=`3082bc39088a535408065976a747b0462b6948a9c35634a6983bcf718c179a9e`，`tests/test_research_ops.py` 当前 SHA=`d094a5a3da4a7b22065474c8e207d2a18e9d3ad5c7133f236002f83d086201e7`。专项=`11 passed`，最新全量=`1262 passed, 1 warning in 132.75s`；规则登记 e4 artifact 后 SHA=`0bcbd4c6090b7af841c7ec2a1d98123ac39b151fcc5f41ba18b8d67638b4b0e6`。
- 这轮证明的是表示层训练稳定性，不是主动探测、Rule-IR 组装、SFT/RL 或 payload 能力。下一阶段仍是第三实现/族外 holdout、ASK/失败动作变化和信息字段消融；accepted training rows 仍为 `0`，不得把 A800 占用或 loss 下降当作晋级理由。

### 2026-08-08 末：PG-333 WebGoat 第三实现与跨实现 A800 表示 smoke

- 新增固定 digest 的 WebGoat disposable relay/collector：`app/pg333_webgoat_docker_relay.py` SHA=`c63356ef7a5264bc2ef8736c9ef2c6a2357bea8b910c5358180e076f4639f1a3`，`scripts/run_pg333_webgoat_typed_get_post_source_rows.py` SHA=`a0d1ee8835f11ca0756d68cedd94049635816f2047386a00cbfa3ec47a62410c`；测试 SHA=`00a90eb6c2734145c1eeb0adbf1a914ed804770091c98fe3b6cad0626a508307` / `ad790f62a71c2abcf6b5246ccc9bc6a1c9c9f55bb77b3d42a42b416887bcc587`。只做 login 页面/重定向 method-shape canary，不做漏洞、认证绕过或 payload 结论。
- WebGoat 3 seeds×2 routes×candidate/reference/negative/replay 完成 24 role/replay episodes、18 source rows；GET/POST 各 1 lane，typed candidate/reference=`6/6`、negative violation=`0`、replay=`6/6`、fresh/reset、network-none、无 bind/volume、context firewall 全通过。报告文件 SHA=`77244d0387002b64ebe65a2af07b1f65db53ed9690d1b520aee3c85b91e50519`，rows SHA=`293dba538678a4496f941c3f9b275bbb8f57e892b03d1c85204b962108a96d25`，source audit SHA=`51bc46a91b8a6b68909282e15dcf766d3cd7cd90668eb9dfe11e1babcf4c980c`，sidecars SHA=`02e2a9ffacaab3a236ecaf8de0c12236c58cc18245ef4af6deeca7f93ade1b2d`；source audit=`passed` 只表示七轴/107-field 行完整，不表示可晋级。
- 与 PG-332 Pikachu/DVWA 诊断集合并为 `research/pg333_three_impl_get_post_diagnostic_source_rows_v1.json`：45 rows、3 implementations、4 families、train=9、implementation holdout=36、unique sequence ratio=`0.466667`、source valid=`45/45`、context firewall=`0`。信息审计=`diagnostic`（文件 SHA=`c535d4f9d1518496f67d7dcd17a3b408b0a7e2a7252e4c40bfc61fca92a8cd6d`，内部 SHA=`2375c702f8d03faac6fb82d0cdd4d81ed1182fdf42c543d824c4030ab9537b4f`），容量 required window=`4145`（legacy 72 仍 FAIL，minimum/balanced 技术可容纳），claimed rows=`30` 但 accepted training rows=`0`。词表 append-only context=`1224`、target=`61`、ontology inventory=`737`，词表外 holdout token=`0`。
- 在授权远程 `112.111.7.91:60228` 的 A800 GPU0、`CUDA_VISIBLE_DEVICES=0` 上做 3 seeds×1 epoch context-only 表示 smoke；训练只读 9 条 train 的 `context_tokens`，holdout 只读 36 条 context，`target_tokens_read=false`。报告文件 SHA=`0613b75ef12c7fbf42425a15742d4992857100236e7c752e901fe120f3280db7`，内部 report SHA=`7617b6b030010a4f37b6584c17d1fa26b8752a420fbe8238e57584b3c340b45c`，checkpoint SHA=`96fccf1c1c293522dddd291b136beb71445f46142fe72a34e8663f3409d72437`；train loss=`7.020527/7.065515/7.034781`，holdout loss=`7.058157/7.073871/7.084190`，holdout predictive entropy=`7.085137/7.086147/7.087068`，无熵塌缩。状态仍为 `representation_pretrain_candidate_only`，不是 capability SFT/RL、Rule-IR 或 payload 训练。
- 研究台现在额外投影 `pg333_cross_impl_diagnostic`：只保留实现/族计数、七轴熵和序列摘要、容量、A800 loss/entropy 与哈希，不输出 records、tokens、wire、payload、response、oracle；`app/research_ops.py` SHA=`3257cd2fd7da65d52377173beb528634ee6b4f50b99dbca9213ac4849a0eb321`，`tests/test_research_ops.py` SHA=`450c412254b7ac894391544358bf8cc272d76c7e7ec2a11230637290b00f9ae1`。PG-333 focused=`4 passed`，全量回归=`1270 passed, 1 warning in 140.64s`；当前本地无 pg333 容器/训练进程，远端 GPU0 post-check=`1 MiB/0%/no compute app`，GPU1–7 未触碰。
- `research/improvement_rules.json` 新增 `framework_first_fast_lane_v1`（大框架先行、按需补细节）和 `stateful_persistent_evaluator_exception_v1`（允许固定 digest、disposable、network-none/loopback 的 stored-XSS/数据库 evaluator 状态写入；不再用一刀切的“持久状态禁止”阻断，但仍禁止公网/外连/真实业务数据/挂载，且 state delta 只留 evaluator-side）；当前规则 SHA=`bba5c7c26f6678d0bddd9fe3bd6728c7d0289e3c7afaafae0a1e82e6b008293f`。快速 smoke 结果仍只能是 diagnostic/ASK，未通过信息、split、typed/fresh/replay、容量或 operator 门时不得训练、记忆或漏洞晋级。
- 下一动作：把 WebGoat method-shape 与 PG-332 stateful GET/POST 继续做族外 ASK/失败动作变化与字段消融；只有出现独立训练 split、信息审计通过且 accepted rows>0，才申请 Rule-IR/SFT/RL 短 smoke。不要因为 A800 空闲重复低信息训练。

### 2026-08-08 末：PG-334 过程 token 诊断与 A800 快速表示 smoke

- 为响应“先搭大框架、动作快、不要被低价值细节拖住”，新增 `scripts/build_pg334_process_token_dataset.py`、`scripts/audit_pg334_process_token_dataset.py`、`scripts/build_pg334_process_token_vocabulary.py` 与 `tests/test_pg334_process_token_dataset.py`。它从 PG-278 controlled loopback fixture 生成 576 条去标识化过程行（train=384、implementation_holdout=192、pre ASK=288、negative=288），移除 family/implementation/slot/raw/oracle 字面量，只保留 method/placement/encoding/reset/未知观测/形状/失败与动作抽象；七轴 manifest、context firewall、pair、ASK、negative abstain、action-change 全通过。数据仍是 controlled fixture，real gold=0，promotion 全部关闭。
- 产物：dataset=`research/pg334_process_token_diagnostic_v1.json`（SHA=`3520bd45c48e0c1340deea5c17785d2acc75f6c26be862dfa5be7a119cd17d1a`），audit=`research/pg334_process_token_diagnostic_audit_v1.json`（SHA=`caf6d4bbb63526a2addf9f7ff7b043fd1f68e9b3361c8f770b6faa354315b8ce`），vocab=`research/pg334_process_token_vocabulary_v1.json`（context=51、target=13，SHA=`f46a1fc5cd727805bd018600f8bac128256ca1ddb45dec5864c28f0a29797072`）。audit status=`diagnostic_only`；unique context sequences=16、target sequences=3，不能当真实网页泛化数据。
- 周末授权远程 `112.111.7.91:60228` 的 A800 GPU0 已执行 3 seeds×1 epoch context-only smoke（`CUDA_VISIBLE_DEVICES=0`、`BLACKBOX_REMOTE_A800_TRAIN=1`）；train=384、holdout=192，`target_tokens_read=false`，holdout unknown token=0。规则哈希更新后已重新锁定并覆盖报告：`research/pg334_a800_process_representation_e1_v1.json`（文件 SHA=`a853fbfa8677eaf0eb66f61ed3eb7a3ad701590e140979524296b40f28919877`，内部 report SHA=`7930b92e1d1e326b6edc046134a4c4a9537b3ae5815b6b9199d0d4a7b1d985f9`），checkpoint=`research/pg334_a800_process_representation_e1_v1.pt`（SHA=`df2812e589405736501230b6eeb1e68f35283e8caa6f133bad481715d222ba67`）；train loss=`3.964983/3.964695/3.992572`，holdout loss=`3.969192/3.965630/3.994416`，holdout predictive entropy=`3.940897/3.946819/3.948515`，无明显熵塌缩。GPU0 运行后复核为 1 MiB、0% utilization、无 compute app，GPU1–7 未触碰。
- 研究台新增 `_pg334_process_token_projection`，只投影行数、ASK/负例计数、词表规模、A800 loss/entropy 与哈希；不输出 context/target tokens、family/slot、wire、payload、response 或 oracle。`app/research_ops.py` 与 `tests/test_research_ops.py` 已加入 PG-334 bounded projection；focused PG-334/PG-333 tests=`6 passed`。规则新增 `pg334_process_token_diagnostic_v1`，当前 `research/improvement_rules.json` SHA=`c0ec9f51b437325072fa89dc12e9bdcfcd93d27cab46d4be2d39b9b781301eb1`。
- 结论：A800 本轮确实训练了“过程 token 表征候选”，不是 capability SFT/RL，也没有读取答案/生成 payload。下一步仍必须接入真实独立实现的 ASK/失败动作变化/阴性 typed rows，完成信息熵/字段消融和 accepted source-row 门；在此之前禁止把 PG-334 fixture 或 A800 表征候选升格为漏洞检测能力。
- 最终核对：PG-334/PG-335/研究台专项通过，最新全量 `python -m pytest -q --durations=10`=`1278 passed, 1 warning in 140.81s`；当前 `app/research_ops.py` SHA=`b41ec6596cb5fb5c4e1c63f512e64ac7a0053d108aba159950a7c057d9ff6e2e`、`tests/test_research_ops.py` SHA=`037c407256919395a6db07ee2065a4d260c2c8a5d22271392510fe7be2a94dd4`。本机没有 PG-334/PG-335/PG-331 训练进程或靶场容器；远端 GPU0 已回到 1 MiB/0%/无 compute app。C/D/E/F 可用约 `81.06/159.48/155.61/89.83 GiB`，本轮没有删除数据；历史大权重仍在 E: 可逆归档、D: junction。

### 2026-08-08 末：PG-335 真实 source-grounded 过程 token 轨道

- 审计发现 PG-333 的 45 条真实 typed source rows 虽完整覆盖七轴、GET/POST、candidate/reference/negative/fresh/evidence，但原始 target 主要都是 `send_probe`，真实缺观测和失败转移几乎为零；直接训练会造成“发送动作高分、不会主动问”的偏差。
- 新增 `scripts/build_pg335_real_process_token_dataset.py`、`scripts/audit_pg335_real_process_token_dataset.py`、`scripts/build_pg335_process_token_vocabulary.py` 与 `tests/test_pg335_real_process_token_dataset.py`。以 45 条真实行作锚点，生成 390 条去标识化过程诊断：observed=45、逐七轴 mask/ASK=315（每轴 45）、failure-repair=15、真实 negative review=15，train/implementation_holdout=`78/312`；mask/failure 是明确 counterfactual diagnostic，不是 real gold。context 只保留抽象网页/请求/响应 token，移除 family/implementation/route/原始 wire/oracle literal；七轴 field manifest、firewall、ASK recall、failure action-change、negative abstain、split isolation 全通过。
- PG-335 数据内部 SHA=`13b0a8c37c5d9f7e170d6879bdd9ed9d3508130cc4cf19c99611485b36c47de4`，audit 内部 SHA=`128ef203b097510887e26a1e285c6637dba03e102e0c803d69882460b59a5855`，vocab 内部 SHA=`9a8c428bebe4aa82217e77075744775fa97a2282863a3db9e48d787219302fee`；七轴 presence 熵均为 `0.515947 bit`，context token entropy=`8.347428 bit`，audit=`diagnostic_only`、promotion 全关。没有把遮蔽样本重标为漏洞答案。
- 当前规则新增 `pg335_real_source_process_token_v1`，规则 SHA=`4d93e0f3ddd59445fef5c455d455af636d69d97eef5b31e380512ff25ffa3eb3`。周末远程 A800 GPU0 已用当前规则哈希完成 3 seeds×1 epoch context-only smoke：train=78、holdout=312、`target_tokens_read=false`；train loss=`5.946902/5.938171/5.947718`，holdout loss=`5.951199/5.947612/5.961948`，holdout entropy=`5.920781/5.923140/5.922582`，无熵塌缩。报告内部 SHA=`2701f8a8ebe1268f413636595c5a60089b8c77801cdbce8d50cb56bf9f3326b5`，文件 SHA=`586574093a7171a900ed6ec2f19767a25aeb9061e27901e4807b385df0add54d`，checkpoint SHA=`09c950df94dfb8d8663885779f3e62bfaf395377bcef9d59b714c3aa257d0fb8`；GPU0 已复核为 1 MiB/0%/无 compute app。
- 研究台新增 `pg335_real_process_token_diagnostic` 投影，边界只展示 source/row/ASK/repair/negative 计数、七轴 mask、熵、A800 loss/holdout，不展示 token、wire、payload、response、oracle。`app/research_ops.py` SHA=`b41ec6596cb5fb5c4e1c63f512e64ac7a0053d108aba159950a7c057d9ff6e2e`，`tests/test_research_ops.py` SHA=`037c407256919395a6db07ee2065a4d260c2c8a5d22271392510fe7be2a94dd4`。该轨道仍是 diagnostic/candidate，不能宣称模型已会发现漏洞或生成 payload；下一步必须采集真实 failure trace，再做 capability SFT/RL。

### 2026-08-08 末：PG-336 真实失败/ASK 过程 token 与 A800 smoke

- 直接复用已完成的 PG-325 授权本地 Docker SQL 回放 trace（固定 digest、network-none、fresh、GET/POST、candidate/reference/negative、typed evidence），没有重复启动靶场。新增 `scripts/build_pg336_real_failure_process_dataset.py`、`scripts/audit_pg336_real_failure_process_dataset.py`、`scripts/build_pg336_real_failure_process_vocabulary.py` 与 `tests/test_pg336_real_failure_process_dataset.py`。
- PG-336 生成 `180` 条抽象过程行：`probe_observed=27`、真实 `failure_repair=9`、真实 `negative_review=9`、真实 ASK preflight=`135`；GET/POST=`120/60`，seed train/holdout=`60/120`。上下文只保留 method、参数角色、编码、历史动作、失败签名、step/replay 和缺失槽位；route/family/implementation/wire/payload/response/oracle/evaluator literal 全部留在 sidecar 或只保 SHA-256。
- audit=`diagnostic_only`，所有检查通过：真实失败动作变化 `9/9`、ASK safe=`0`、阴性复核 abstain=`9/9`、context firewall=`0`、七轴 manifest 存在、GET/POST 存在；明确 `independent_implementation_holdout=false`，accepted training rows=`0`，promotion 全关闭。数据 SHA=`5be086df122fd796006494f3f54c3001a450b842a70d12f184f0a2c363429040`，audit SHA=`1e3a5c553fca7ef57ed2981bbfb7b5c78ad9d84cb2a6066a004fe86efedf90d1`，vocab SHA=`d881d3d548949e04cda792b72a6d3147df7e61e1f811c0d0c3e403bdd0cfabc4`；专项数据集测试=`4 passed`。
- 新增 `scripts/run_pg336_a800_real_failure_representation_smoke.py`，只读 `context_tokens`，seed holdout 仅作诊断，显式 `CUDA_VISIBLE_DEVICES=0`、`BLACKBOX_REMOTE_A800_TRAIN=1`，3 seeds×1 epoch。远程主机 `112.111.7.91:60228` GPU0 为 `NVIDIA A800-SXM4-80GB`；train=`60`、seed holdout=`120`、target tokens 未读取、词表外 token=`0`。train loss=`3.563286–3.643402`，holdout loss=`3.579385–3.653609`，holdout predictive entropy=`3.611522–3.618483`，无明显熵塌缩。
- PG-336 A800 报告=`research/pg336_a800_real_failure_representation_e1_v1.json`，文件 SHA=`925808e226025552bf2991fb2ce9d62a94cdefa85a3b0b10b18bde8f7f9c4dfe`，内部 report SHA=`c1d0af2a0f601da4af130c71b77a2dc557101486712f2cd8fae094f2553a67e0`；checkpoint SHA=`8897d8f0ef718d8a208933a1d7a1570f23549c80eea7753274ac67746f03e937`。GPU0 运行后复核为 `1 MiB/0%/无 compute app`，GPU1–7 未触碰。
- 研究台新增 `pg336_real_failure_process_token_diagnostic` 投影；只显示失败/ASK/阴性/GET/POST计数、熵、seed holdout 和 A800摘要，不输出 records、tokens、wire、payload、response、oracle。`app/research_ops.py` SHA=`00b16375fa3a646003a300b450446c4893e2737fa5592cc80244a14a780e1154`，`tests/test_research_ops.py` SHA=`f1decb3d13352404e6ab0c2bbf638b6ee0a65701b04b37f56f46f5ef2d535b57`，PG-336/研究台专项=`9 passed`。
- 规则新增 `pg336_real_failure_process_token_v1`，当前 `research/improvement_rules.json` SHA=`c9cad72166305e127f5ba7cce1537b2d0061c9b3e31d57584b91270836c8da21`。本轮结论：A800 已实际训练表示层候选，但 capability SFT/RL、长期记忆和 payload/vulnerability claim 仍关闭；下一步是第二独立实现的真实 failure/ASK/negative 轨迹与字段/熵消融，不是重复低信息训练。
- PG-336 接入后的全量回归：`python -m pytest -q`=`1287 passed, 1 warning in 145.69s`；唯一 warning 仍为 Torch nested-tensor。A800 runner=`scripts/run_pg336_a800_real_failure_representation_smoke.py` SHA=`1e20a9728fbc74d29431efef04777ac7960fc2480849294b5f946c43a5d1fb0b`，runner tests=`tests/test_pg336_a800_representation_smoke.py` SHA=`67f35dc6bf27f47742aa10a24fd6d6935ebB6d29cd7fdc8ee43b91b105fc394a`；C/D/E/F 当前可用约 `81.12/158.62/155.61/89.83 GiB`；PG147–PG216 历史权重已在 E: 可逆归档、D: junction，当前 PG-331/332/335/336 证据未移动。

### 2026-08-08：PG-337 独立实现 failure→repair→abstain 与 A800 表征 smoke

- 新增 `scripts/run_pg337_dvwa_failure_repair_replay.py`，只在授权的固定 DVWA digest disposable 容器中运行：每个 candidate/reference/replay fresh target 先发无效果 POST、GET 观察失败、再发单变量修复 POST；negative 在失败观察后 abstain。network none、loopback relay、无挂载/端口、数据库 clean、teardown 和 role-bound evidence SHA 全部硬门；原始值/响应/标记只在 evaluator 内存，不进入 source row。seed `33701` 实跑 `status=completed_failure_repair_diagnostic_only`：typed candidate/reference/replay=`1/1/1`，negative violation=`0`，failure observed/action-changed=`3/3`，repair observed=`2/2`，fresh/db/teardown=`true`，source rows=`3`，training/promotion/memory/payload/vulnerability 全为 `false`。报告内部 SHA=`bbf9b433700c9cf74bc72cc3a7edd80cb07d826151b15e6e106f0ede43dad2e7`；文件 SHA：report=`92bd1f2effc10742b2decb9e67bfa2f520cc04688acddfbd8e79fc688a3846ac`，rows=`6531c04c98087f22105da056768c2b0e327930325be2c3aea87444f2c0d16e80`，sidecars=`7eb13f1f01e64a296aec04fa891ffa2de091eb734ea0d6d092d5b09ea315b8bc`。
- `scripts/build_pg337_cross_impl_process_token_dataset.py` 合并 PG-336 与真实 DVWA 过程行，形成 `183` 条抽象 process-token rows（train=`60`，implementation holdout=`123`，failure-repair=`11`，negative=`10`，ASK=`135`）；`scripts/audit_pg337_cross_impl_process_token_dataset.py` 通过 context firewall、split、failure target、negative abstain 和 hash checks，但因尚未人工晋级保持 `diagnostic_only`、accepted training rows=`0`。数据/审计/词表文件 SHA 分别为 `e951f60ddcb99c815ab30a2b6d976a1288877e8509412ec7ce5e0add21495233`、`ff0588726e577c4a2d6955b66c62e9488fd69c07b664836722eab0f4d9db0e24`、`243b45092ba7bd05ec678fc5315c3b511a52db8c02634f7368b33861aa459539`；内部 dataset/audit/vocab SHA 仍由各 artifact 字段绑定，不能手改。
- `scripts/run_pg337_a800_cross_impl_representation_smoke.py` 已在远程 `112.111.7.91:60228` 的 `NVIDIA A800-SXM4-80GB GPU0` 运行（周末、`CUDA_VISIBLE_DEVICES=0`、`BLACKBOX_REMOTE_A800_TRAIN=1`）。它只读取 context tokens，target tokens read=`false`，3 seeds×8 epochs，train=`60`、holdout=`123`，required window=`16`，unknown token=`0`；train loss=`3.328517/3.327948/3.424503`，holdout loss=`3.358351/3.362062/3.446524`，holdout entropy=`3.714777/3.708592/3.721276`，未发现 entropy collapse，但 `information_promotion_gate_passed=false`、status=`representation_pretrain_candidate_only`。报告内部 SHA=`63139254ecaac4b563daaa1889f13f38f441cbd9b24049240c6b78c7bd200205`；本地 report=`0e6936d477d6ea52882188235f7cf8f9edaef424fc24a08601ba6e04f56f5e94`，checkpoint=`fd2053f2b6a52b74bf7328b59bd31b3e21b547b590e52504ae57fce6afe36e97`。训练后 GPU0 只读复核为 `1 MiB/0%/no compute apps`，GPU1–7 未触碰。
- 研究台 `app/research_ops.py` 已加入 `pg337_cross_impl_process_token_diagnostic` 聚合投影，并优先展示最新 e2 A800 候选；只展示 count/entropy/split/holdout/A800/hash，不输出 records/tokens/wire/payload/response/oracle。`app/research_ops.py` SHA=`35be01c990f606da12e7f969ef7c7417563b5afcd2f1fc1ac8eab6fe1ca6c1a7`，`tests/test_research_ops.py` SHA=`3ccaec31da99c64e1fb422bcbd73ae822e53e069927ba7cb8eab71f750ef1417`。规则 `research/improvement_rules.json` 已登记快速大框架路径，但仍要求先做字段/熵消融、第二族外实现和人工复核；规则 SHA=`6a9673a86aefb140a25124e329616409f9314ec3bcec47bdcb340d875829e0c8`。PG-337 focused tests=`7 passed`，研究台专项=`3 passed`；完整 capability SFT/RL、长期记忆和漏洞能力声明仍关闭。
- PG-337 接入后的全量回归：`python -m pytest -q --durations=10`=`1297 passed, 1 warning in 138.29s`；唯一 warning 仍为 Torch nested-tensor。该结果只证明工程合同/fixture 通过，不改变 `diagnostic_only`、`accepted_training_rows=0` 或任何 promotion gate。
- 本轮新增 e2 后，4 个已结束历史目录 `pg152-real-mix-architecture-v1`、`pg181-manifest-decoder-v1`、`pg190-dual-head-action-gate-v1`、`pg258-unified-rule-ir-capacity-v1` 已可逆移至 `E:/blackboxanalyze-archive/artifacts/legacy-large-20260808`，原路径保留 junction；`research/storage_archive_manifest_v1.json` SHA=`6c2d644e12b53492425de13e6d1ff94a2e690050fb971f3cfbe06fe4b3f650e2`，未删除任何文件。PG-259/260/261 与当前 PG-331/332/335/336/337 证据未移动；当前 C/D/E/F 可用约 `79.74/158.81/155.13/89.83 GiB`。

### 2026-08-08：PG-338 full-axis information-preserving process track

- PG-337 三 seed DVWA fresh replay 已完成：9 source rows，failure observed/action-changed=`9/9`，repair observed=`6/6`，typed candidate/reference/replay=`3/3/3`，negative violation=`0`，fresh reset/database clean/teardown 全部通过；仍为 evaluator diagnostic，training/promotion/memory/payload/vulnerability 全关闭。
- 新增 `scripts/build_pg338_information_preserving_process_dataset.py`、`scripts/audit_pg338_information_preserving_process_dataset.py`、`scripts/build_pg338_information_preserving_vocabulary.py`。PG-338 合并 WebGoat 18 条完整 source rows 与 DVWA 9 条完整 source rows：`27` 行，WebGoat train=`18`、DVWA implementation holdout=`9`，failure-repair=`6`、negative=`3`；每行保留 7 轴整页抽象 context，长度 `508–1358`，七轴覆盖率 `100%`，字段消融每轴 changed rate=`1.0`，context-target alignment=`1.0`，firewall forbidden=`0`。审计明确 `accepted_training_rows=0`；document 轴本批 entropy=`0`，因此仍需更多页面形态，不能把完整字段误判成充分泛化。数据/审计/词表文件 SHA 分别为 `0596bda8e0ac41d015ab1499dc506c58611eb8d17a789befeb847c61010837cd`、`ec18e0a43879d361756aae56da8caa948003036f2482f6008e3ab6ea72b176aa`、`bf33ba107c67b26bdf883cb00611662afcd576ff0f938d56bf9b312f5487a48f`。
- `scripts/run_pg338_a800_information_preserving_representation_smoke.py` 已在远程 `112.111.7.91:60228` 的 A800 GPU0 运行：3 seeds×2 epochs，context-only，target_tokens_read=`false`，required window=`1358`，train/holdout=`18/9`。train loss=`6.030984/5.985079/6.005866`，holdout loss=`6.022700/5.947719/6.006833`，holdout entropy=`6.063813/6.066691/6.064485`；表征候选可运行，但信息晋级、能力 SFT/RL、长期记忆和漏洞/payload 声明仍关闭。报告内部 SHA=`103921006a75fbc8806d7008b8b0922eb359b5c68d286f95ffbbae0be37cbd44`；本地 report=`b84b8af5a0f12cdb8346d4a1d439a139deda82ff90f2d654c56591cdff87ed16`，checkpoint=`0b1484d0b8bb7119ea36480e7dcc809644c3c6c9e39ecd7ba5e650efced50a56`。runner SHA=`c25e60dd9f16635dc206d309be9f8cace98e5046b518f5f2c42f6f01b18ae451`。
- 研究台新增 `pg338_information_preserving_process_token_diagnostic`，只投影 axis entropy/coverage、split、context length、A800 loss/entropy 和 hash；不输出 records/tokens/wire/payload/response/oracle。`app/research_ops.py` SHA=`0af44ac32b33302385e91e359263b83cfcc63254dddcec4b2456f6facfeb2247`，`tests/test_research_ops.py` SHA=`9c1375f8ecafbe29dd4ca9d6ab00a57f5426af3f72d634de946c2e1507957291`，PG-338 tests=`3 passed`，PG-337/338 UI专项=`8 passed`。规则 `research/improvement_rules.json` SHA=`2da03de638d3082f0903cf4521733ab296de7e4923975986fccbdaeb30a72e75`；下一步仍是补页面形态/轴熵与做受控 capability SFT 前的 hard-negative/ASK holdout。
- PG-338 接入后的全量回归：`python -m pytest -q --durations=10`=`1302 passed, 1 warning in 146.44s`；唯一 warning 仍为 Torch nested-tensor。收盘核对：本地 running containers=`none`，远端 A800 GPU0=`1 MiB/0%/no compute apps`；C/D/E/F 可用约 `79.98/161.80/155.13/89.83 GiB`。

### 2026-08-08：PG-339 多页面形态 full-axis holdout 与 A800 表征 smoke

- 审计 PG-333 三实现整页 source rows（45 条）与 PG-338 full-axis rows（27 条）后，新增 `scripts/build_pg339_multi_shape_dataset.py`、`scripts/audit_pg339_multi_shape_dataset.py` 和 `tests/test_pg339_multi_shape_dataset.py`。构建器保留原始 `source_split`，把 implementation holdout 显式改名为 `shape_holdout`，按抽象 context+target 去重且 holdout 优先；source implementation 只存哈希 sidecar，raw payload/response/oracle/evaluator 不进 context。结果：输入 72、去重后 24，train=9、shape_holdout=15、duplicates=48、rejected=0、accepted_training_rows=0；数据 SHA=`28380d1f0b3596c7f70ad6b8bbb1f13dc3e5be2bf7afed5fdf5daf8218338b03`，builder SHA=`fed14240278877962ece45cd3a885103bb94d2e4fcf484131b59ac84a42fdb04`，audit SHA=`1492cf9e45a490be036ecbf44e9a4d13d4fc496c21fe894e93571cd198184ba1`，dataset audit SHA=`f16a2dd7cf0876df7e055b783469b666ac2894b184f2c4e363e3bf55e3bad8fc`。七轴 presence/field status 与字段消融均记录，split/implementation isolation 通过；预测熵尚未作为晋级门，audit 仍 diagnostic-only。
- 新增 append-only 词表合并器 `scripts/build_pg339_multi_shape_vocabulary.py`，只合并已冻结的 PG-333/PG-338 context manifests，不从 shape holdout 拟合；context vocab=1238、holdout_rows_used_for_vocabulary=false、forbidden=0，词表 SHA=`c59093061cdd3dfb2f921907e600f55ff8d1b1068560be2f6b0117f09cb0fc4d`，脚本 SHA=`1801bae5aadfcca93056fee567240cf5f56bf5c93cb2f70b0618d2fbee5bbfd2`。
- 周末授权远程 A800 GPU0 已完成 `scripts/run_pg339_a800_multi_shape_representation_smoke.py`（SHA=`35de08dcc13b2ad7483032381715e0879728a9e2a9b39adb67b50fc963348a80`）：`CUDA_VISIBLE_DEVICES=0`、`BLACKBOX_REMOTE_A800_TRAIN=1`、3 seeds×1 epoch、train=9、shape holdout=15、required window=3284、`target_tokens_read=false`、词表外 token=0。报告内部 SHA=`df041c91085d5f2de3fa2c6d7d5a143cf2e89f6836f4c3081d7998f4b6c5ead9`，文件 SHA=`6f7bccd068cee26a3112141cd594f84d47668e40be868ccd063ec571ac348b6d`，checkpoint SHA=`8c827e60a1c4359f872507073b2a5456e545e224369290fa66456c7ee6b9956c`；train loss=`6.999332/7.096776/7.024460`，shape-holdout loss=`7.056247/7.104404/7.054356`，holdout predictive entropy=`7.095692/7.096646/7.097789`。所有 promotion false；这只是 context-only 表征候选，不是 Rule-IR/SFT/RL、payload 或漏洞能力。
- 研究台新增 `_pg339_multi_shape_projection` 与 bounded model/metric `pg339_multi_shape_information_preserving_diagnostic`，不输出 records/tokens/wire/payload/response/oracle；`app/research_ops.py` SHA=`3f41c035a42c81289dbe5b856db23a6b8cde016812bd09575b1ab96af72a0458`，`tests/test_research_ops.py` SHA=`88225d1216f1965a009c70ab8876a50a22fb74c123570cbf5a38736647fd5bef`，PG-339/338 focused=`10 passed`；规则 `research/improvement_rules.json` 已登记 `pg339_multi_shape_information_preserving_v1`，规则 SHA=`079c9c2ce8e16ec51ad04a9c182058f0453e68471b9ff548c45ce3c91ea366ac`。
- 全量回归：`python -m pytest -q --durations=10`=`1310 passed, 1 warning in 144.85s`；唯一 warning 仍为既有 Torch nested-tensor。训练后核对：远端 GPU0=`1 MiB/0%/no compute app`，GPU1–7 未触碰；本地 `docker ps` 为空。下一步不是重复训练，而是对 shape holdout 跑预测熵/字段消融的正式门；只有门通过并有 accepted source rows，才申请 Rule-IR/SFT/RL。任何失败保留 diagnostic，不写长期记忆。

### 2026-08-08：PG-339 训练前后熵基线与逐轴消融（e3）

- PG-339 runner 现在在同一 seed 下先测未训练模型，再测训练后模型，并对 shape holdout 逐一移除 `axis_begin/axis_end` 区间；不读取 target token。runner SHA=`c9e5780bc77c0a1eb328cbdabd33ad596d4c57fcc0324667fdd8c317995ce6f3`，测试 SHA=`517c2d690a684aef40a28f88dfaf859e1c83f1c361eda2bd74164975a78c16f0`。
- e3 报告=`research/pg339_a800_multi_shape_representation_e3_v1.json`，文件 SHA=`d6e1a94c7fbdb657814df3d6f75e074b9de52a3b87bc8d4cc37ffa063d72e0e2`，内部 report SHA=`4e7a39765ce297cd12b45f095d080da9a335d7781e0f2defe85beff8efe23e65`，checkpoint SHA=`2a50a69e4507a24937623a2f8ebe17f2bf30d5330b18fe4e8beb3b6db00f556e`。3 seed 的 baseline→post 最大相对熵变化=`3e-06`（25% 门通过），shape holdout 熵约 `7.095692–7.097789`；document 轴消融 loss 增加约 `0.065–0.101`，navigation/request/response/JavaScript/failure/belief 消融大多接近零。
- 解释：熵没有塌缩是好消息，但逐轴结果说明当前 train=9 的样本主要让模型使用 document 轴，其他轴尚未形成可验证依赖；因此不能因 entropy gate 通过就进入 Rule-IR/SFT/RL。必须先扩大不同 transport/JS/failure/belief 的真实 source rows，且保持 implementation/shape holdout。
- e3 已绑定最终规则 SHA=`9df4f5c0051304911852d24d5a73fe251677e3d203392b7a71f9aca046c6f870`；研究台优先 e3，`app/research_ops.py` SHA=`723c49d08232acf3e1855c750a7ae027a4dbfa9ca36031959486d8baaee406b9`，`tests/test_research_ops.py` SHA=`b8fe0d308303644f2fdf6c955c5c7cd07669b3b1071da35985be762618d997c2`。promotion 仍全 false，下一步是增加真实多轴训练行和主动 ASK/失败修复 holdout，不是调大模型。

### 2026-08-08：PG-339 e5 规则锁定、快速框架规则与 stateful evaluator 例外

- 规则新增/明确 `framework_first_fast_lane_v1`：先打通 schema/adapter/runner、抽象 token、Rule-IR slot、GET/POST 形状和 context-only smoke；只把不影响下一动作的细节延后。授权范围、network-none/loopback、live fresh reset、candidate/reference/negative、typed evidence SHA、context firewall、split 隔离和容量不截断硬门不可跳过；延后的细节只能标记 `diagnostic/unknown/incomplete`，不能伪装成训练 gold、漏洞确认或 payload 晋级。
- 规则新增/明确 `stateful_persistent_evaluator_exception_v1`：固定 digest、用户授权、disposable `network=none`/loopback 靶场可以测试 stored-XSS 或数据库 challenge state 的持久类 POST。每 seed×route×role/replay 独立容器，reset 前后、database-clean attestation、baseline、candidate/reference/negative/replay typed evaluator、role-bound evidence SHA、state delta 只留 evaluator-side、teardown/reset；禁止公网、外连、真实业务数据、凭据、bind/volume 和 raw payload/response/state 进入模型上下文。旧的 read-only lane 仍保持各自 no-stateful-write 合同。
- PG-339 e5 使用当前规则和归档清单锁定：规则 SHA=`e1d83cb6f7388346fae8f5a8ea7cd09ccc9bac42c486c51b2c3b31b32f681d1c`；报告=`research/pg339_a800_multi_shape_representation_e5_v1.json`，文件 SHA=`268005f2e8baa01708a981e659d5dc08afd8cd2b8fc6fb352ea564514f0ce451`，内部 report SHA=`6fea7ee27916380d95af8360b6a9b573694473a45db1203191a7d0763564c934`；checkpoint=`research/pg339_a800_multi_shape_representation_e5_v1.pt`，SHA=`50607bd9f71e04a0ed90a264b80d340dc198344e42f40f415873eb2ec60ecdaf`。e5 为远程周末 A800 GPU0、`CUDA_VISIBLE_DEVICES=0`、3 seeds×1 epoch、context-only、train=9/shape_holdout=15、target_tokens_read=false；最大 baseline→post predictive-entropy 相对变化=`3e-06`，熵门诊断通过但 `information_promotion_gate_passed=false`，所有 promotion 仍 false。逐轴消融依旧显示 document 轴有可见依赖，transport/JS/failure/belief 轴依赖不足，不启动 capability SFT/RL。
- 研究台已优先 e5：`app/research_ops.py` 当前改动后 SHA=`387dc8f76ecc9d6f6f6a40937e018e7294b34a08c8184cc9744382dab4d2dc73`，`tests/test_research_ops.py` SHA=`a4b6aa5e3eeb6f71b3e89715f819400c0ba13946289cf3e097dfc7b9b0b9c391`；PG-332/PG-339 rules/UI focused=`64 passed`，全量 `python -m pytest -q --durations=10`=`1314 passed, 1 warning in 151.68s`。唯一 warning 仍为 Torch nested-tensor。
- 存储治理 v4 已复核：25/25 个明确历史目录位于 `E:/blackboxanalyze-archive/artifacts/legacy-pre-pg217-20260808`，D: 原路径均为 junction，总计 `10,688,629,389` bytes；未删除、未触碰 PG259+/PG331+/PG332+。清单=`research/storage_archive_manifest_v4.json`，SHA=`11f2d9b562599b58e295bd06efda9e609f64b252635251066c92d25ea1d3bb9c`；归档后 C/D/E/F 可用约 `79.877/161.392/155.132/89.832 GiB`。

### 2026-08-08：PG-339 e6 最终规则/研究台锁定

- 规则将 PG-339 最新报告预注册为 e6，当前规则 SHA=`bad7cb876fcf3aed137888d342a7b3bc5c57a650c1640e6f4b402271373076a8`；e6 报告=`research/pg339_a800_multi_shape_representation_e6_v1.json`，文件 SHA=`f1d915a09f695839edc228e85a9e5fde7816da249812a366823d9475a6db1718`，内部 report SHA=`07a01f7676db88856a364c71b0fd518c2aaa63fb462518059d7317e1dc137049`；checkpoint SHA=`1a0ed1633d1d9082e94de54217c839421cf11e94203a29a6ea58ce8ef5f47164`。e6 锁中的 rules/dataset/audit/vocab/script/model 全部与本地一致，A800 GPU0、`CUDA_VISIBLE_DEVICES=0`、3 seeds×1 epoch、context-only、train=9/shape_holdout=15、target_tokens_read=false；最大 baseline→post predictive-entropy 相对变化=`3e-06`，信息晋级仍 false，promotion 全部 false。
- 研究台优先 e6：`app/research_ops.py` SHA=`c7061de6e3683bc320c5028002ee24a36001923a08479c0e41c24b1f671e556c`，`tests/test_research_ops.py` SHA=`b85cb2a2a7a0276e7cd6999b61777f4c8789799572163d88436906163d29205f`；规则/PG-339/研究台专项=`64 passed`；全量 `python -m pytest -q --durations=10`=`1314 passed, 1 warning in 148.94s`，唯一 warning 仍是 Torch nested-tensor。规则回归测试已从固定 e3 改为“latest/e6”断言，SHA=`5f1adb36eb39acb46bdb521e92867e1700ff59df205c4448dbe89c36338f8885`。
- e6 后置资源核对：远端 A800 GPU0=`1 MiB/0%/无 compute app`，GPU1–7 未查询/未触碰；本地 `docker ps` 为空；无 PG-339 训练进程。后续不再重复低信息 context-only smoke，下一科学动作是补充能改变 navigation/request/response/JavaScript/failure/belief 消融的真实 source rows，再申请 capability ASK/失败修复轨道。

### 2026-08-08：PG-340 平衡实现分割与多轴表征 smoke

- PG-339 的轴消融定位出明确数据问题：9 条 train row 的 `response_transport` 只有 1 种序列，模型主要使用 document 轴。新增 `scripts/build_pg340_balanced_axis_dataset.py`、`scripts/audit_pg340_balanced_axis_dataset.py`、`scripts/build_pg340_balanced_axis_vocabulary.py` 与 `tests/test_pg340_balanced_axis_dataset.py`。PG-340 从冻结 PG-333 source rows 构造新的表示分割：Pikachu+WebGoat 训练、DVWA implementation holdout；原始 split 保留为 `source_split`，实现标签只留单向哈希 sidecar，context 不含 implementation/family/route/wire/payload/response/oracle。
- PG-340 数据：输入 45、去重后 21，train=15、shape/implementation holdout=6，训练实现=2、留出实现=1、重复=24、rejected=0、accepted_training_rows=0；实现哈希交集=0、context firewall=0。数据/审计/词表文件 SHA 分别为 `f132ac3b575a21d6a2d52577281823225fd6f872e97896e2cf28ea52d75a8602`、`5bae4af2add96b8fe1f8cbb71ae0f686214c7d10f9e7b61dbd81998e56332657`、`68e6f09a8709a49af9e1f74eec69a883d45f21df7694f82d4bb87cc61f256b8a`；专项数据/审计/词表测试=`3 passed`。这是一条诊断表征数据，不是 capability gold。
- PG-340 已在周末授权远程 A800 GPU0 做 3 seed×1 epoch context-only smoke，使用当前规则 SHA=`384f5970993ec54e34045d31c3e1c015cab0547b97466119db1af24ed4210bd1`，train=15、holdout=6、required window=3284、target_tokens_read=false、unknown context token=0。报告=`research/pg340_a800_balanced_axis_representation_e1_v1.json` SHA=`8192a82b68128863ee7a0d26ce4e8b2ed1ca92d551fb15c40d58f372539a2d44`，checkpoint SHA=`505771ea1338a212d5dd2993a2d00d3065509e9451a90eb261bb235507238f87`，内部 report SHA=`b3075c562f6dbf1ce35993c7da45d6e9e366b5cc121bd4069775d31a5f12087e`。熵相对变化最大=`0.0`，熵门诊断通过；但逐轴消融仍只有 document loss delta 明显（约 `0.067–0.116`），transport/JS/failure/belief 依赖仍接近零。结论是“分割更合理但数据量/目标仍不足”，不启动 capability SFT/RL，不晋级记忆或 payload。
- 研究台已加入 bounded PG-340 projection `pg340_balanced_axis_representation_diagnostic`，只显示数量、实现分割、轴熵/消融摘要、A800 loss/entropy/hash；`app/research_ops.py` SHA=`f7d399319486e9f7dc0d1163aa69cbf28cd5b7de10a25a33dc9e81500efa6ad4`，`tests/test_research_ops.py` SHA=`90c77dc5cabc0fecade37c8805602cb42cb8fdff36b83354399a1d2fe5c2de30`；PG-340/PG-339/研究台专项=`67 passed`。规则已登记 `pg340_balanced_axis_representation_v1`，当前规则 SHA=`384f5970993ec54e34045d31c3e1c015cab0547b97466119db1af24ed4210bd1`。
- PG-340 接入后的全量回归：`python -m pytest -q --durations=10`=`1319 passed, 1 warning in 147.19s`；唯一 warning 仍为既有 Torch nested-tensor。训练后远端 GPU0 已恢复空闲，本地没有容器或 PG-340 进程；下一步是建立 target-conditioned ASK/失败修复数据审计，不把当前 context-only 候选误当成能力模型。

### 2026-08-08：PG-341 目标条件解码两轨与 A800 诊断

- PG-340 的缺口已被验证为“目标没有和完整七轴上下文对齐”：PG-338 full-axis train 的 18 行全部 `question=none/next_action=select_probe_variant`，ASK/repair/abstain 只在 implementation holdout；PG-337 coarse process 则有 183 行、train/holdout=`60/123`，两边都包含 ASK、repair、negative-abstain。不能把两种表示拼成一套 gold。
- 新增两轨构建/审计/词表：`scripts/build_pg341_target_conditioned_dataset.py`、`scripts/audit_pg341_target_conditioned_dataset.py`、`scripts/build_pg341_target_conditioned_vocabulary.py`、`tests/test_pg341_target_conditioned_dataset.py`。产物 `research/pg341_target_conditioned_process_full_axis_dataset_v1.json` 共 210 行（coarse=`183`、full-axis=`27`，coarse train/holdout=`60/123`，full-axis train/holdout=`18/9`），所有 row `training_eligible=false`；coarse 只允许独立 target-decoder diagnostic，full-axis target training 明确 false。数据/审计/词表文件 SHA 分别为 `ac05b76de5faeb1808460d1e5ca911f8f2929f75c48e7b37df90de7b32ed5c94`、`d79509b034f4f9174178fc95c4d38ae3015b29bcbcebf643e075b7101abd0192`、`74ae619d0cd994a2cf96f258bc4c30a28091c6dae5e2e86e8c825abbb838c9c8`；audit=`blocked_full_axis_target_gap`，coarse diagnostic flag=`true`，full-axis target coverage=`false`。构建/审计/词表脚本 SHA=`a08ab98ade58901db6dbf41bebcc1fc989b15c191c8d2f578dd084a9346fdbfb`、`193fd7a13979ff995a306894b1454058c196205748f570a0a82e1c00c17b1b08`、`94274c3e9c05743e80fd1724455a271ba069ea767db26703efbd406ed391063d`；专项数据测试=`3 passed`。
- 新增 `scripts/run_pg341_a800_target_conditioned_smoke.py` 与 `tests/test_pg341_a800_target_conditioned_smoke.py`。runner 只读 coarse process 抽象 context+target，使用分离 context/target inventory、3 seeds、目标权重对照，并把所有 seed state 写入 checkpoint；full-axis 不会被偷换成训练数据，promotion/memory/payload/vulnerability 全 false。runner SHA=`1c4fe89a4c6d04c0406501003cab76586ef0495353ccb229054b9d89bfd761d6`，测试 SHA=`e44abe41f5cff4cc8e45320d70094af5d6bc7ca6991371e1e13bee1a0ffd17b3`。
- 周末远程 `112.111.7.91:60228`、`NVIDIA A800-SXM4-80GB GPU0`、显式 `CUDA_VISIBLE_DEVICES=0` 完成四个预注册 candidate-only 对照，GPU1–7 未触碰，运行后 GPU0 均恢复 `1 MiB/0%/无 compute app`：
  - e1：1 epoch、目标权重 2、holdout ASK recall=`0–0.058`，几乎未学会目标解码；报告文件 SHA=`fe475f2bf2c0fa45045e207c3a8ee6ce383cfa695db1f6230cb91ca9ea691fb9`，checkpoint=`a40e770e6ec3a8a7be96ca5aded5ce6eb67f32908f97cfa72cf1f0a19561c278`。
  - e2：8 epoch、目标权重 2，holdout ASK recall=`0.404–0.865`，仍全部安全拒绝；报告 SHA=`5931d58cbeec60f9ed3ab7e3d5f86aded88e8bfd94251f57d36c90d689393786`。
  - e3：32 epoch、target/context=`5/0.25`，holdout ASK recall=`0.865–0.885`、sequence exact≈`0.732`，但 positive recall=`0`，说明模型学会了问/拒绝，却没有学会正向 Rule-IR 组装；报告 SHA=`2674c45bd046930e7456a61ac2491b3a9a13e9eb6763c4123355d54502f90355`，checkpoint=`059aa75a53979f256216c47e3dfdbe4f40eda179e0d687c22270e8fac7cd47fe`。
  - e4：在 e3 基础上给 `safe_to_send=1/assemble_rule_ir` 正类权重 12；最好 seed holdout positive recall=`0.111`，但 hard-negative false-allow=`6`，其余 seed 仍为 0，故严格门失败；报告 SHA=`3d51deb7604ffd3ed1877372cb09a3d456020dbc9e11bfbc4f63ecdff6299e84`，checkpoint=`d21e1f36ffb858052eaf872f6a1ae3bb81d4481593472df0ac930012fb8e229e`。
- 解释：A800 训练确实执行了，但 e4 证明“简单正类加权”会引入负对照误放，不能用 loss/ASK 高分掩盖；当前模型只能作为 coarse target-decoder diagnostic，不能称为整页漏洞检测、Rule-IR capability 或 payload 生成模型。下一科学动作是采集至少一个独立实现的完整七轴 failure→repair/negative source rows，使 full-axis train 与 holdout 同时包含 ASK、repair、abstain，再重做信息熵/字段消融和目标条件训练；不再盲目堆 epoch。
- 规则新增 `research_execution_policy_v1.pg341_target_conditioned_two_view_v1`，当前 `research/improvement_rules.json` SHA=`a060c77e1d33feb541e34154cd3f07493e1fcaf7bf80d56f8e2a39f778d8b12b`，预注册 e1/e2/e3/e4 权重对照与 full-axis target coverage 硬门。研究台新增 bounded `pg341_target_conditioned_two_view_diagnostic` projection：只显示两轨计数、目标覆盖、A800 aggregate 指标和 promotion flags，不输出 records/tokens/sidecars/raw。
- 研究台改动 SHA=`a31346c4b0bc31f3941b20e695bef350ef4abc0d665b2fe254cb174306fade53`，`tests/test_research_ops.py` SHA=`9019b6250364086707cf33b416d359a9eb515eb9b3a7679f50cec2aa69ae0e6d`；PG-341/PG-340/PG-339/规则专项回归=`14 passed`。随后全量 `python -m pytest -q --durations=10`=`1326 passed, 1 warning in 199.03s`；warning 仍为既有 Torch nested-tensor。当前本地无容器/训练进程，远端 A800 GPU0 已释放；C/D/E/F 约 `69.4/161.2/155.1/89.8 GiB` 可用；PG-341 当前 artifacts 未归档。

### 2026-08-08 22:40：PG-342 全轴失败修复采集合同与快速路径收口

- PG-341 e4 的 A800 目标条件训练确实执行过，但最好 seed 的 full/implementation holdout positive recall 仅=`0.111`，hard-negative false-allow=`6`；其余 seed positive recall=`0`。因此不能把“ASK 学会了”解释成 Rule-IR 组装或漏洞能力，当前 accepted training rows 仍为 `0`。
- 当前资源只读核对：远端 `112.111.7.91:60228` GPU0=`NVIDIA A800-SXM4-80GB`、`1 MiB/0%`、无 compute app；GPU1–7 未查询/未触碰；本地 `docker ps` 为空。A800 空闲不代表可以绕过数据门。
- 新增 planning-only `scripts/plan_pg342_full_axis_failure_repair.py` 与 `tests/test_pg342_full_axis_failure_repair_plan.py`。计划固定 3 seed×3 implementation（2 train、1 implementation_holdout）×GET/POST×candidate/reference/negative/replay，共 18 episodes；每 episode 要求七轴/107-field manifest、fresh role identity、typed evidence SHA、baseline→失败观察→动作改变→修复、阴性 abstain。计划不启动 Docker/网络、不生成训练行；artifact=`research/pg342_full_axis_failure_repair_plan_v1.json`，SHA=`dbb86aa223b5c924e6b78e386a84359f48e34802e26095382f705fb642b721b9`；planner SHA=`a41833f93590004722eb380dc28e9d2b6220256dfe3743a03840378c7af6d044`，tests SHA=`bd5c56ddc0be13fc1fd0f46a04128a76226b571286a951d61ee7cd9ab3bfb75f`。
- `research/improvement_rules.json` 已登记 `pg342_full_axis_failure_repair_plan_v1`，并保留 `framework_first_fast_lane_v1` 与 `stateful_persistent_evaluator_exception_v1`：可先搭大框架、延后不改变下一动作的细节；但授权、network-none/loopback、fresh/reset、GET/POST 真值、candidate/reference/negative、typed evidence、context firewall、split 与容量硬门永不跳过。规则当前 SHA=`f709f5762fdd797bfdc137179871230ec1ed785cfa08ca520a89ef822345d9db`；规则专项+PG-342=`9 passed`。
- 下一安全动作：只接入一个已授权的 reviewed live adapter，先完成 PG-342 真实 full-axis failure/repair/negative rows，再做 source/implementation split、信息熵/字段消融；在 full-axis train 与 holdout 两边同时有 ASK/repair/abstain 且 audit 通过前，不再重复 A800 capability 训练。快速 smoke 只允许输出 `diagnostic/unknown/incomplete`，不产生 payload/漏洞/长期记忆晋级。

### 2026-08-08：PG-342 WebGoat 真实回放、全轴数据与 A800 表征诊断

- 已在固定授权 WebGoat disposable lane 完成 1 seed×2 路由（GET/POST），每路 candidate/reference/negative/replay 独立 fresh，failure→action change→repair、typed positive、negative clean、证据与清理门通过；源回放报告=`research/pg342_webgoat_failure_repair_report_v1.json`，文件 SHA=`2b2459db9e9dbb0b01dc4d821b84288093eff71218bf5fa97ea64b6fc540f67a`，source rows=`research/pg342_webgoat_failure_repair_source_rows_v1.json`（SHA=`6b7e687c8e803e4974743093764fb705b0ea3e0fe33b08272daea0762993e4ba`），evaluator sidecars SHA=`1ef838986ab4778fb6e705b2bc4da1ad1a6739b22229c8055b18a2fcd2d47ac8`。
- PG-342 与 DVWA implementation holdout 合并为完整七轴诊断数据：`research/pg342_full_axis_failure_repair_dataset_v1.json`，15 rows（train=6、implementation holdout=9、GET=10、POST=5），七轴均存在、context-target alignment=1.0、context firewall=0；dataset SHA=`92f03949b820411ebe6484ccd110906a765441dd158f9e5eb5977ab3751dfb02`。审计=`research/pg342_full_axis_failure_repair_audit_v1.json`，SHA=`5a51c85e1318a1c72fb240a8d308bf815cca0c4adbabee205a3708b6f40e47fe`，状态 `diagnostic_only`，科学 gate 仍 blocked、accepted training rows=0。词表=`research/pg342_full_axis_failure_repair_vocabulary_v1.json`，SHA=`1adca2395c7cd6c9a2fe6ad19fd24390722a3360ea76d92a5c73d48ea839845f`，context=432、target=14。
- 新增 context-only runner=`scripts/run_pg342_a800_full_axis_representation_smoke.py`，SHA=`b0afb5d5653c1513783f7a80c33d2631c39c63ea8b682430fa1a5f31a4c9b5e8`；专项测试=`tests/test_pg342_a800_full_axis_representation_smoke.py`，与 PG-342 回放/计划合计=`10 passed`。runner 只读 abstract `context_tokens`，`target_tokens_read=false`，不把 raw payload/response/evaluator sidecar 放入模型或 checkpoint。
- 周末远程 A800 GPU0 v2 已完成：远端 `112.111.7.91:60228`、`NVIDIA A800-SXM4-80GB`、`CUDA_VISIBLE_DEVICES=0`、3 seeds×1 epoch；train=6、implementation holdout=9、required window=1358、词表外 token=0。报告=`research/pg342_a800_full_axis_representation_smoke_v2.json`，文件 SHA=`c5308dc8346827063c688751ab8bbf8ca065fb254b1a144fe71e1bbd40d06991`，内部 report SHA=`ee75f2658964a47b7d9e76eb681de5c1dca9f3f536fdcae7078bb24c419761eb`；checkpoint=`research/pg342_a800_full_axis_representation_smoke_v2.pt`，SHA=`6478c845f332be0459ac795b9ef24a53782cf512738e706fe8cb56d1f405d5b2`。最大 baseline→post predictive entropy 相对下降=`0.000062`，熵门通过；document 轴 loss delta 约 `0.060–0.096`，navigation/transport/response/JavaScript/failure/belief 轴大多接近零或不稳定，说明数据仍不足以证明多轴真正被模型使用。
- v2 锁定数据/审计/词表/规则/代码/模型哈希；运行时规则 SHA=`b59632480961069ffa4e630abbac405bdf1fe78f94839561b1bb4593caba3cb7`。运行后 GPU0=`1 MiB/0%/无 compute app`，本地 `docker ps` 为空。规则已登记 `pg342_full_axis_failure_repair_live_diagnostic_v1`；promotion、model memory、payload catalog、vulnerability claim 全部 false。
- 研究台已增加 bounded `pg342_full_axis_failure_repair_diagnostic` projection，只显示 counts/gates/loss/entropy/hash，不显示 records/tokens/sidecars/raw；`app/research_ops.py` SHA=`1ce4d5d42f609cfb9b314a8996a62d74ae051d8df78c3fdc185f2b78bd51c403`，`tests/test_research_ops.py` SHA=`337d3c35ff834ad247514eea70be47a9be25a0406d69a46c428298e966bf588c`，PG-342/PG-341 专项=`4 passed`，全 `tests/test_research_ops.py`=`61 passed`。
- 结论：A800 确实训练了“表示候选”，但这不是 capability 模型。下一动作是增加能改变非 document 轴的第二/第三独立实现 failure→repair/negative rows，再做多 seed 目标条件 ASK/Rule-IR 训练；不因熵门通过而晋级长期记忆。长期记忆只写本文件，遵循 `long_term_memory_memo_policy_v1`。

### 2026-08-08：PG-343 目标条件上下文绑定审计（阻断）

- 目标：在任何 ASK/Rule-IR SFT、offline RL 或 A800 训练前，确认完整七轴上下文能唯一决定抽象目标；特别检查 candidate/reference/negative、failure/repair 的角色和步骤是否已经进入上下文。
- 完成：新增只读审计器 `scripts/audit_pg343_full_axis_target_conditioned.py` 与测试 `tests/test_pg343_full_axis_target_conditioned.py`；输入为 PG-338、PG-339、PG-342 的抽象记录，去重只按 `context+target`，不输出 token、sidecar、payload、响应或 evaluator 答案。
- 结果：输入 66 行，去重后 30 个 context-target，24 个唯一上下文；训练侧 21 行、实现留出 9 行；目标覆盖已经同时出现 probe、ASK、repair、negative-abstain，但有 6 组相同上下文对应多个 target hash。审计状态=`blocked_role_step_context_missing`，不是训练轮数不足。
- 证据：`research/pg343_full_axis_target_conditioned_audit_v1.json` SHA=`7d49e313cc122b9e76cd9886a93bfeb1e4725da2b5f7ba8b753c7050cc32372a`；审计器 SHA=`f808f711230dd7c682193bbb292e09350c358172ef96a521137e3ff16bc92cf8`；专项测试 SHA=`b7e324c667ab7273b07b609d14677523c562e5f53b6268177f0fb55d00429958`；规则文件已登记 `research_execution_policy_v1.pg343_full_axis_target_conditioned_context_binding_audit_v1`，当前 SHA=`7d31dad90da2656aaa5c6838264e6861fd6cbe011c28981bf2fd9ce44697fc42`。
- 失败原因：采集上下文缺少 `role_step_abstract_token`，导致同一输入可对应候选/参考/阴性或不同动作；若直接训练会把冲突标签压成随机 next-token 映射，并可能放大误报。
- 下一安全动作：在新鲜授权回放中补入抽象 role/step token、candidate/reference/negative 绑定和 failure→repair 步骤绑定，重新生成 source rows；重新通过 context 唯一性、split、字段熵/消融和容量审计后，才可申请短 A800 target-conditioned smoke。PG-343 当前不生成训练行、不启动 Docker/GPU、不提升长期记忆或 payload catalog。

### 2026-08-09：PG-343 role/step 绑定回放与全轴目标条件数据

- 角色绑定实现：新增 `app/pg343_role_step_binding.py`，只接受 source/evaluator 明确 attestation 的 `candidate/reference/negative/replay` 与 `preflight/baseline/failure/repair/replay`，输出抽象 `belief_probe_role`、`belief_process_step`；禁止从 target token 反推角色，并拒绝 raw/route/oracle/target literal。模块 SHA=`716fe0cdace6f03936f13fdf95b4ac659aa3e21c3c2fc159d17b0e225fb5d8bf`；tokenizer SHA=`586396e0833c08b555c43283dcccc7f6228d9f1a213e946770b2de6eb2a271d4`；专项 binding 测试 SHA=`019a05b57f7e9a8ee430ebd399038040d6dd2168b4012ccc3756afc9e0207aaa`。
- 新鲜回放：PG‑342 runner 接入显式 role/step binding，使用 seed `34301` 完成 1 个 GET + 1 个 POST；candidate/reference typed=`2/2`，negative violation=`0`，failure action-change=`2/2`，repair=`2/2`，每角色 fresh/network-none/loopback/清理均通过，source row failure 为空，training rows=`0`。版本化证据：report=`research/pg343_webgoat_role_step_binding_report_v1.json` SHA=`ca8405725319ca4fbcfe9628748a1874446f178094abc3f15e83f3780062dc4d`；source rows SHA=`a09faa3fec352f8fe49581c13a86f97a84416c04246fa2a39ef9e160e3937b09`；sidecars SHA=`579145233f0a634a4f89a0ddccadbbb500f8625754ed8b432582e6a0cd5f293e`；runner SHA=`c55d5e65f1f002cbc0470ce3143b547c5b4a8925ba7fd3f9f5f8375165af3130`。
- 证据命名修正：旧固定路径 `research/pg342_webgoat_failure_repair_*_v1.json` 在本轮被新 seed 写入，不能再冒充先前 seed 的不可变证据；当前新 seed 已复制到上述 `pg343_webgoat_role_step_binding_*_v1.json` 版本化路径。旧 PG‑342 哈希只保留为历史记录，当前验证一律使用版本化文件，后续 runner 不得覆盖固定 v1。
- PG‑343 dataset builder/audit：新增 `scripts/build_pg343_role_bound_dataset.py`（SHA=`10a069750cf502fca71329e3bafbe7473304a6003934574c02f20b6a35e2d5cd`）与 `scripts/audit_pg343_full_axis_target_conditioned.py`（SHA=`914646b94ba02f20be054ed8fc24db35ff943cc6f7180cbf2dcbf7057c7c867d`）。33 条输入去重后 21 条（train=15、implementation holdout=6），12 条历史 context-target 重复被保留为计数；上下文唯一性冲突=`0`、split/source leakage=`0`；七轴 token sequence unique=`document 17 / navigation 11 / request 8 / response 3 / JavaScript 9 / failure 10 / belief 12`。dataset SHA=`9558fff1c0c70c831b5d1a282b07af81453b01b90c23ed33550ce1086d4b62db`；audit SHA=`db83edc391e831e4db776ba8cd1da01773b3ecd7b4807c89c3b52bc002f37f8e`，status=`diagnostic_passed_not_training_eligible`，promotion 全部 false。
- 规则：`research/improvement_rules.json` 已登记 `pg343_role_bound_full_axis_target_conditioned_dataset_v1`，当前规则 SHA=`4721e07ad9081f84ca8475eab4bdc79724649419b2f26339700576c111a85c83`。PG‑343 专项（binding、builder、audit）=`8 passed`；当前尚未运行目标条件 A800。
- 下一安全动作：锁定以上 dataset/audit/代码/规则哈希，实现只读 target-conditioned ASK/repair/negative runner；先做短 A800 GPU0 candidate smoke，保持 `training_eligible=0`、promotion/memory/payload/vulnerability 全 false，验证最坏 seed、目标序列准确率、ASK/repair/negative 和 predictive entropy 后再决定是否扩充采集。

### 长期记忆与核心模型身份（持续有效）

- 项目逻辑上的核心模型只有一个：`decoder-only causal Transformer/MoE + Rule-IR tokenizer/adapter`。PG-327～PG-343 的 checkpoint、seed、熵正则或目标权重对照都是同一架构的候选实验，不是多个已晋级模型；除非单独通过全部硬门，否则一律标记 `candidate/diagnostic`。
- 长期记忆唯一写入位置是本文件 `AGENTS.md`；聊天内容、checkpoint、研究台投影和 payload/evaluator catalog 都不是长期记忆。写入只追加可审计事实、失败原因、证据 SHA-256、资源清理状态和下一安全动作，不改变 promotion。
- PG-343 target-conditioned A800 GPU0 candidate smoke 已完成；以后每轮运行前仍必须锁定 dataset/audit/vocabulary/rules/script/model 哈希，运行后记录最坏 seed、`variant_recall`、ASK/repair/abstain、negative false-allow、熵变化和 GPU0 释放状态。当前下一动作是补独立实现的完整七轴 ASK/repair/abstain/variant rows；任何门失败都保持 `blocked/diagnostic`，不得写入训练晋级或长期记忆结论。

### 2026-08-09：PG-343 target-conditioned A800 candidate smoke 结果

- 远端 `112.111.7.91:60228` 的 `NVIDIA A800-SXM4-80GB GPU0` 实际完成 3 seed（`34311/34312/34313`）×8 epoch、`CUDA_VISIBLE_DEVICES=0`、最大上下文 `3296`；训练前所有 gate 通过，训练后 GPU0=`1 MiB/0%/no compute app`，本地无 Docker 目标。
- 证据：report=`research/pg343_a800_target_conditioned_smoke_v1.json`，文件 SHA=`1670dc338d75a24490c94aec4f133b6486a718614e6a734c34da8b89d1c7aacb`，内部 report SHA=`e0dc3fa28dfad8157a2d9655d537fba0efcbae97da8d57dab7272cd8fe90a086`；checkpoint=`research/pg343_a800_target_conditioned_smoke_v1.pt`，SHA=`b2e65f1248029277bb9b8d3c02d729529cf542d5f30276e6038c59bc0d9a64b7`。
- 结果：最大 holdout predictive-entropy 相对下降=`0.000405`，负对照误放=`0`；最坏 seed 的 ASK/repair/abstain/`variant_recall`/positive recall 均=`0`。因此这是“锁门通过、训练实际运行但目标解码未学会”的 candidate-only diagnostic，不是 Rule-IR、漏洞检测或 payload 能力。
- 本轮修复了 runner 工程 bug：`_load_rows` 原先校验 role/step 后丢弃绑定，导致 gate 假性阻断；修复后的 runner SHA=`94be749511fa0db57d6c9606019b8b95dfc63cd4ef95fe9cc14c797effde873d`，专项测试=`2 passed`。预注册规则 SHA=`1191731e07b3a7af1fbd38d1b4972b70ad9b9275352eae19560692a1cb231ade`；包含结果台账后的当前规则 SHA=`5095960f1762cb88f226ad2a8d0881af0e630779981dd8d6075b0660014842fc`。
- 下一安全动作：不要用加 epoch 或放宽 negative 门补分；先补至少一个独立实现的完整七轴 ASK/repair/abstain/variant source rows，再做 target-conditioned 训练。promotion、长期记忆晋级、payload catalog、vulnerability claim 继续全 false。

### 2026-08-09：PG-343 后置验证

- 全量回归 `python -m pytest -q --durations=10`=`1349 passed, 1 warning in 176.10s`；warning 仍是既有 Torch nested-tensor。PG-343 专项与规则专项已先通过 `16 passed`。
- 训练后资源复核：本地 `docker ps` 为空、无 PG-343 训练进程；C/D/E/F 可用约 `68.3/160.9/153.6/89.8 GiB`。历史大目录仍按可恢复 junction 归档，未做删除或覆盖。

### 2026-08-09：PG-343 target-loss v2/v3/v4 对照

- v2（`34321–34323`、32 epoch、target/context=`12/0.25`）报告=`research/pg343_a800_target_loss_ablation_v2.json`，文件 SHA=`cebfdcb9a678860978faeb0d1f4a5441717692df1df9ddcbb39d5d8d4bf61346`，checkpoint SHA=`22ceeff1d3fd0263f6e6ab9ce3f512e2f262e12f972b64a1bef1a3d58797992d`；holdout ASK/repair/abstain/variant recall 全=`0`，最大熵下降=`0.003202`，negative false-allow=`0`。提高 target 权重不是解决方案。
- v3（`34331–34333`、32 epoch、target-only、lr=`1e-4`）报告=`research/pg343_a800_target_only_sft_v3.json`，文件 SHA=`0e086d8e24a30214f4094ce448ed8a8639bea10c100b5fa79c7960627f6664ca`，checkpoint SHA=`1f71a90a8769ccff51a2f0652df0570bc7fb24ca968bd0000ce54993648e6ac0`；holdout 动作 recall 全=`0`，最大熵下降=`0.003239`，negative false-allow=`0`。仅对 target 计算 loss 仍不足。
- v4（`34341–34343`、32 epoch、target-only、lr=`1e-3`）报告=`research/pg343_a800_lr1e3_target_sft_v4.json`，文件 SHA=`c30a58c75e69532cb7cf17fc6e7c75087921ad03dc59c2c929adcab836ab9134`，checkpoint SHA=`189c7dac761e5f12ba732eed8e1b9058a220a6cdd09515bd32a1e5db6c2e5bf9`；训练集部分 recall 提升，但实现留出 recall 全=`0`，最大熵下降=`0.510505`（43.8%–51.1%），超过 25% 信息门，明确 blocked。高学习率只放大记忆化/熵塌缩，不能晋级。
- 三轮均在远端 A800 GPU0 实际完成，结束后 GPU0=`1 MiB/0%/no compute app`；promotion、长期记忆、payload catalog、vulnerability claim 全部 false。当前规则台账已登记 `pg343_a800_target_loss_ablation_results_v2_v4`，规则最新 SHA=`df82107af8899d284b92dc04cd2641163f321a5424e5a4cd948b4007730f069e`。
- 研究结论：PG‑343 的问题不是简单 target loss 权重，而是少量长序列上的上下文—目标映射/跨实现结构不可学习；下一步应添加显式 decision boundary 或跨位置摘要机制，并增加独立实现的同形目标行，再做小规模可学习性测试；不要继续盲目加 epoch/学习率。

### 2026-08-09：长期记忆唯一载体再次确认

- 规则 `research/improvement_rules.json` 的 `research_execution_policy_v1.long_term_memory_memo_policy_v1` 明确：长期记忆唯一写入 `AGENTS.md`；启动新一轮工作、上下文压缩、阶段停机和用户要求记忆时，必须先读本文件，并把可审计摘要追加回本文件。
- 聊天上下文、模型权重/checkpoint、研究台投影、payload/evaluator catalog 都不是长期记忆；它们只能作为待核对的证据或候选状态，不能替代本备忘录。写入备忘录不会使模型、payload 或漏洞结论晋级。
- 本次规则补充了写入触发条件、单一来源约束和遗忘保护：若某结论没有写入 `AGENTS.md`，就视为尚未持久化；下一轮必须先恢复本文件再继续。规则文件本轮 SHA-256=`c4a884bb731019b98ffe53cbb72c052bda88614a4ad28ba4e24f9b2953d56051`；AGENTS.md 自身哈希只在文件外核对，不写入自身。

### 2026-08-09：PG-344/345 数据分层与目标解码诊断

- PG-344 修复了 role/step 构建器对 versioned typed-method/typed-stored-post collector 的漏收：60 条已采集抽象输入保留 30 条唯一 context+target；重新按一向实现哈希划分为 train=21、implementation_holdout=9，实现交叉泄漏=0、上下文冲突=0、七轴序列仍有变化。dataset=`research/pg344_cross_impl_role_bound_dataset_v1.json` 文件 SHA=`258f69e0b264530b8ef0651dc71ad2a420b8e7977c11b3ef454b532bf4fd5998`；audit=`research/pg344_cross_impl_role_bound_audit_v1.json` 文件 SHA=`e1ed45e6f9503ba8dabddc74c032703b1688a68c668314471cda3c69af2da91a`；vocabulary=`research/pg344_cross_impl_role_bound_vocabulary_v1.json` 文件 SHA=`a33cd003b371a878cecf7c9eb77c60b44bf753047e29488e95c0bb0509ca445a`。仍 `training_allowed=false`。
- PG-344 A800 GPU0 3 seed×16 epoch、target/context=`2/0.25` 完成；熵下降最大=`0.000999`，negative false-allow=`0`，贪心 train/holdout ASK、repair、abstain、variant、positive 全为 0；teacher-forcing train/holdout 也为 0。report=`research/pg344_a800_cross_impl_smoke_v1.json` 文件 SHA=`1b71856935d1f7a918ea358574219a45474b928757e7078881f6c33cd0f6f0a8`，checkpoint SHA=`d48e02d77132759074099faabd8fb98ba6f2d916c0005d8cd8244773d665af16`。
- PG-345 只在 context 末尾追加抽象 `decision_boundary=target`，不删除任何七轴 token；16 epoch、归一化权重损失 v2 仍 teacher-forcing/贪心为 0，说明边界 token和简单损失修复不是根因。32 epoch、lr=`1e-3` 的 v3 训练 teacher-forcing 提升到约=`0.26455–0.603175`，但实现留出最高仅=`0.061728`，贪心目标仍全 0，最坏熵下降=`0.263404` 超过 25% 门，明确过拟合/blocked。v3 report 文件 SHA=`6120372ae46cda44d75eabc257ceaa98bfdf40975754f06339c100163e98c676`，checkpoint SHA=`fbe5356c846e79744045b373673d745e5edb4861e975d919cf2d27c1f4ac21b8`；规则已登记 `pg344_pg345_a800_target_decode_results_v1_v3`。
- 结论：当前模型不能安全或自主发出“正常 payload 测试”。它仍是抽象 Rule-IR candidate 诊断，尚未生成稳定的抽象 target，更没有任意 wire 能力；明天若展示，只能展示 teacher/evaluator 辅助的授权 loopback 流程，明确标注模型未通过 capability gate，不得把人工参考 wire 冒充模型能力。下一动作是独立 target-slot/结构化解码（保留 causal next-token 表示层）和更多同形跨实现 source rows，不再继续盲目堆学习率。
- PG-346（预注册，候选尚未运行）：新增 `app/pg346_structured_target_slot.py` 的固定 Rule-IR slot heads，读取 context-only boundary hidden state；causal next-token backbone 与完整七轴上下文保持不变，不把 target 喂入 slot 推理。数据沿用 PG-345 implementation-disjoint 21/9 split，规则登记 `research_execution_policy_v1.pg346_structured_target_slot_decoder_v1`，promotion 全关闭。运行前 SHA：module=`59b1808408fb8cb630bdfa0aee018c8ebc6ec9450b8ed51f0c193b5be2d3e870`，runner=`c911c0b955a088f871d526eb57de40dd0fbaa985696034917b3b9c5b4f43f042`，backbone=`63ddcdc80846e63aa5caf981723660c746011429fc16ef80eb1e03991c649b5b`；下一安全动作是在周末授权 A800 GPU0 做 3-seed candidate-only smoke，并独立审计 ASK/repair/negative/熵门。
- 工程改动：`app/pg295_causal_moe.py` 新增可选 `normalize_weighted_loss` 与 `forward_hidden`；PG-343 runner 当前 SHA=`38a211b2ddf93a614f962388c4107cb332cf641287184a022a12f800911510b1`；规则当前 SHA=`395bb37c3ad92ed22a435ce087e7dbb7a3337e122576437de1b54171cb8d3a71`。
- 资源：PG-345 v3 后远端 GPU0=`1 MiB/0%/no compute app`，本地无运行容器/训练进程；当前 C/D/E/F 可用约=`67.74/160.34/153.56/89.80 GiB`。仍保留所有被规则/审计引用的报告与 checkpoint；历史大目录继续使用 E 盘可恢复 junction，不删除证据文件。

### 2026-08-09：PG-346 结构化 target-slot A800 候选

- 运行：远程 `112.111.7.91:60228` 的 `NVIDIA A800-SXM4-80GB GPU0`，显式 `CUDA_VISIBLE_DEVICES=0`，3 seeds=`34601,34602,34603`、16 epoch、lr=`2e-4`；未启动 Docker/网络/靶场。report=`research/pg346_a800_structured_target_slot_smoke_v1.json` 文件 SHA=`343a11bdf87d7116f129d60426e73d310647a113cb6d92d9ba18fd8292eeb1b1`，internal SHA=`8b51470139a06206a81866355f532ad878d3a8d1d5b6a6a08ad76a46b3b6dfac`；checkpoint SHA=`801d19d7ec9e99824c9345c1a4d7bf0aa761a957a65eb4daec39155a077614a0`。
- 结构：`app/pg346_structured_target_slot.py` 的 slot heads 只读 context-only boundary hidden state；完整七轴 context 和 causal LM loss 保留，target 只作训练标签。训练集 slot 部分拟合（question/safety 最高=`1.0`），但 implementation holdout 的 `next_action=0`、ASK=`0`、repair=`0`，阴性误放=`9/9`；最坏熵下降=`0.001241`，熵门通过但能力门失败。独立 audit=`research/pg346_a800_structured_target_slot_audit_v1.json`，文件 SHA=`90ec700ec1aacd96b0edc624c726c68b0620f5b37d404e6b739d56993f744c7a`，status=`passed_observation_blocked_capability`。
- 结论：结构化解码解决了“长序列目标槽位不可见”的一部分工程问题，但没有解决跨实现条件泛化；`training/memory/payload/vulnerability` promotion 全部 false，不能发出或绑定 payload。下一安全动作是补同形跨实现的失败/ASK/negative 轨迹并审计条件信息，而不是继续加学习率；明日只能展示 evaluator-assisted loopback。
- 本轮新增 SHA：audit script=`454c254ff0f211a2005f7c4c186939874537e5942a37b5aac921fac6b7115d45`；当前规则文件 SHA=`c3f9767e409b680f7350bdb6d3f951d03b64cc98aa845e6f00ca5a232b69e904`。远端训练进程已结束，GPU0 应回到空闲，容器仍为零。

### 2026-08-09：PG-347 多实现 full-slot A800 候选（能力门阻断）

- 修复了 PG-347 runner 的审计路径兼容性：`run_pg343_a800_target_conditioned_smoke.py` 现在同时读取顶层和 `audit.counts.axis_token_sequence_entropy`；缺失仍 fail-closed。新增 nested-axis regression 后，PG-347 gate 才能真实反映数据，而不是被 runner shape bug 假阻断。
- PG-347 数据：`research/pg347_multi_impl_full_slot_dataset_v1.json`（39 行，train=27、implementation_holdout=12、5 implementation groups、split/context/source leakage=0、训练 eligible=0）；audit=`research/pg347_multi_impl_full_slot_audit_v1.json`，seven-axis unique sequences 最低为 3，独立实现 attestation 仍 false。保持 diagnostic-only，不把旧 rows 伪装为 gold。
- 周末远程 A800 GPU0 实际完成 3 seeds=`34701,34702,34703`×16 epoch、lr=`2e-4`、context-only slot inference；训练前 gate 全通过，未启动 Docker/网络/靶场。report=`research/pg347_a800_multi_impl_slot_smoke_v1.json` 文件 SHA=`a2ede69e28b236b95168a9d0f4354cf3057a599599e093444277313e21976b8d`、内部 SHA=`791adcd38621fe90b912a86957867f4eb5483158f29291c9ff69f212925131bd`；checkpoint=`artifacts/pg347_a800_multi_impl_slot_candidate.pt` SHA=`bb9d047911dd24714d7f0d23d0b620e348ecc8f2b921a546c3c92016421037eb`；独立 audit=`research/pg347_a800_multi_impl_slot_audit_v1.json` SHA=`4a6f4f52d6c20d8234b18310c5f3eacbdab737073b32898ca5f0e0218a7967b2`，status=`passed_observation_blocked_capability`。
- PG-347 结果：train slot 部分拟合（ASK/repair/negative 在 train 可见），implementation holdout 的 ASK=`0`、repair=`0`、negative false-allow=`12/12`，variant=`1.0`，最大 predictive-entropy drop=`0.00243`。因此模型仍不会在族外实现上安全提问/修复，promotion、长期记忆、payload catalog、vulnerability claim 全部 false；这不是 payload 能力证明。
- 新增独立审计器 `scripts/audit_pg347_multi_impl_slot.py`（SHA=`ea30725d9be8bb89b285ba6077524ef456030bc72ba5340775a5e07c52208834`）和审计测试；PG-343 gate regression + PG-346/347 专项回归合计通过。当前规则哈希（结果登记后）需以文件外核对为准，不能写入自身。

### 2026-08-09：PG-348 五百实例本地页面/表面注册表

- 为“至少 500 个网址”建立了可复现的本地 fixture registry：`fixtures/pg348/registry_v1.json` 共 `520` 个独立 challenge instance、`520` 个 challenge ID、`520` 个 source hash、`160` 个机制/表面家族、`30` 个模板、GET/POST=`310/210`、14 种 transport+encoding 组合，重复 ID/hash=0，外网记录=0，业务状态写入=0。它们是异构的合成安全表面，不冒充 520 种理论漏洞；真正漏洞效果仍需独立 typed evaluator。
- pages_a：120 静态页面，10 模板×12 变体；manifest SHA=`a76cf3c7bf49ba8e96eb766f8e2a82921e31815f6d17abc0eb8bc2ccb8dc01a5`。pages_b：120 静态页面，GET/POST=`60/60`、10 模板/12 机制；manifest 文件 SHA=`9411fad3e40eed1fdb6c2ad113698579aa70366cc2edb6c9daab8373151d0274`，generator SHA=`52b2e354c2c6ac76526679b0d9693ffbc10d68e1630a1b98d09a501623e95617`。pages_c：本地生成器输出 280 个页面，脚本 `scripts/generate_pg348_500_local_fixture_registry.py` SHA=`19c1f8256be45a900eca6b5e410e66673fe2aef378ef6ab30e03116801fc1f1a`。
- registry SHA=`4d81bf4516602b0994f886742777ed45983da3a2132b37d2735cd79ee0557142`；所有原始 HTML 仅 evaluator-side，`training_context_raw=false`，不含 raw payload/response/oracle/凭据/外联。`app/pg348_surface_projection.py`（SHA=`1f7948cf5c2f16baa3e7d4cf6901cd3ee66e1aa08fc99d7818757b170b340236`）已把 520 条记录投影为七轴抽象 token；缺字段显式 not_observed 并安全 ASK，520/520 projection clean，promotion 全关。
- `research/improvement_rules.json` 已登记 PG-348 registry 和 framework-first fast lane：可先跑 schema/adapter/抽象 token/GET-POST wiring 的大框架 smoke，低价值细节按触发条件补；但 authorization、loopback/network-none、fresh reset、正/参考/负对照、typed evidence、context firewall、split isolation、capacity 等硬门永不跳过。长期记忆只追加到本文件；当前规则文件 SHA=`0a98960d2cd3bf8541cf7ae7557ae98b836f02231ce925ffbbf9064696b38522`。
- PG-348 focused tests（fixture registry + surface projection + PG-343 nested gate + PG-347 audit）=`21 passed`；A800 训练后远端 GPU0 应为 1 MiB/0%/无 compute app，本地无靶场容器。下一步是对 520 条页面先做 source/template/implementation disjoint split、字段/轴熵与容量审计，再决定是否 context-only representation smoke；不能因为数量达到 500 就直接训练或声称检测漏洞。
- 最终回归：`python -m pytest -q --durations=10`=`1391 passed, 1 warning in 172.62s`；warning 仍为既有 Torch nested-tensor。回归后本地 `docker ps` 为空，远端 A800 GPU0=`1 MiB/0%/无 compute app`，GPU1–7 未触碰。规则文件外部核对 SHA 仍为=`0a98960d2cd3bf8541cf7ae7557ae98b836f02231ce925ffbbf9064696b38522`，AGENTS.md 作为唯一长期记忆载体已追加本轮证据。

### 2026-08-09：PG-348 动态运行时修正与 context-only 诊断

- 用户指出静态页面不能复现运行时行为；PG-348 规则已修正：520 个静态页只作 surface/token 预训练 fixture，任何漏洞/能力 claim 必须接入动态 loopback runtime。新增 `app/pg348_dynamic_runtime.py`（SHA=`ebea0c54b332fdb5df0ce4880b2f5274041dc9245ef7c62a40b0e56e46bac35a`）和 `tests/test_pg348_dynamic_runtime.py`（SHA=`5dc4fd923884bb4dc1d91da6eef7dea08e5636c612c7dbe5a89ca547647b8f50`）：只绑定 loopback，动态 GET/POST/302/内存 ephemeral-state，输入不回显，不写数据库/文件，不连外网；fresh reset 与 typed evaluator 仍是能力硬门。
- `scripts/build_pg348_context_dataset.py`（SHA=`9c9b1b73159b5ffbff04ed370d60e4aebd367c48143d4cccf409082914a093af`）读取页面只在 evaluator-side，输出 520 条抽象 context rows（train=240、implementation_holdout=280、training eligible=0）；dataset=`research/pg348_context_only_dataset_v1.json` SHA=`b963fa4bbe4bdbbe6acf7ad090bab1a9f0f94ae2737f0b2437062c12089f03fb`，vocab=`research/pg348_context_only_vocabulary_v1.json` SHA=`e3f2f70885b39557979f93ceedec3157efe2e9cb5972800982778b5da498679c`，audit=`research/pg348_context_only_information_audit_v1.json` SHA=`000083bcc25821ddc9ee2ade37a455a0ba7e6c0c25b6820112fbe258ad475988`。
- context 信息审计显示 document/navigation/request/response/JavaScript unique sequences=`38/5/52/3/17`，failure/belief=`1/1`（尚未观察，必须 ASK），审计 status=`diagnostic_only`，failures=`missing_typed_evaluator, missing_failure_and_belief_observation, synthetic_fixture_only`。因此没有启动静态 context-only A800 训练，避免把页面数量当作能力；先补动态采集和失败/belief/typed rows。
- 动态 runtime focused=`3 passed`；PG-348 当前规则文件 SHA=`352648be0586819cc8e44462a8b4c8b1b0b8d11dcbdb5b59587ed574cb294372`。下一安全动作：在同一 520 registry 上为每个动态 lane 逐 seed×route×candidate/reference/negative/replay 做 fresh reset、GET/POST 真实动态响应投影、failure→repair 和 evidence hash，再重新做字段/熵/容量审计；未完成前不训练 capability、不存长期记忆、不生成 payload gold。
- 规则文件外部最终复核：上一行记录的是动态规则追加前的中间哈希；当前 `research/improvement_rules.json` SHA=`948861c514ea0ed72bfcb617d168a7716b40937453635613560e751a3d260da7`。AGENTS.md 自身不把自身哈希写入内容。
- 动态修正后的全量回归：`python -m pytest -q --durations=10`=`1396 passed, 1 warning in 176.76s`；动态 runtime、PG-348 registry/context/projection、PG-347 audit 和历史工程合同均通过。回归后本地 `docker ps` 为空，远端 GPU0=`1 MiB/0%/无 compute app`，GPU1–7 未触碰。
- 动态 runtime 追加全 registry 检查后，`tests/test_pg348_dynamic_runtime.py`=`4 passed`，覆盖 520/520 records 的 fresh reset、GET/POST、输入不回显和 ephemeral state；测试 SHA=`00b3f8ece01eb8ba4e70e59da4a2e3819e84fe2dab4251ce532c4c36667e82b7`。规则文件最终外部 SHA=`55b725fe28fdbb1f629d7aabf15bd36d325cfb7d8cd23fdd41d6a6880687e2ef`。
- 动态 shape collector `scripts/collect_pg348_dynamic_shapes.py` 已实际遍历 520 条 registry × candidate/reference/negative/replay，生成 `research/pg348_dynamic_shape_trace_v1.json`：2080 行，GET/POST=`1240/840`，fresh reset=`2080/2080`；只保存 status/content/body-length/redirect/input/state 抽象投影，不保存 body/input。trace SHA=`40fd5d625acb69dd29e81f9db63123cab6f92940130cbfd071342c883bbec907`，collector SHA=`05003f114e2af0bf0a6ee84d16fb23dd32bf1b1a521b0ab2e5b6a5d918e25a97`；typed positive=`0`，training eligible=`0`，失败原因是 typed evaluator 未接线和 candidate/reference/negative effect 未确认。动态 shape tests=`2 passed`，当前规则最终 SHA=`09730111b0da9f593bf5e0eed67e77a4c228226139f646f7ebfec0c3eb2be3da`。
- runtime 并发死锁风险已修复：`DynamicFixtureApplication._state` 不再持锁递归 reset；动态专项现为 `6 passed`（含 520 条 direct runtime rows），更新后的 runtime SHA=`5e5f0bd1f88eaa422701d2085a845617a30da89bc3b0d757b8f359b4a4063b29`，规则文件最终 SHA=`0490e5f69e994a1e617f21dbb2fc24029ffebc9eb6ffc2f83f353f9c5fa6483f`。

### 2026-08-09：PG-348 动态 context 表征候选（能力门仍阻断）

- 动态 runtime 已不是静态占位：`app/pg348_dynamic_runtime.py` 对 520 个 registry 实例提供 loopback-only 的 GET/POST/302/内存 ephemeral-state 行为；`scripts/collect_pg348_dynamic_shapes.py` 以 candidate/reference/negative/replay 四角色采集 2080 行抽象响应 trace。运行时不保存原始 body/input，不写数据库/文件，不连外网；fresh reset=2080/2080，但 typed positive=0、training eligible=0，原因是 typed evaluator 与 effect 对照尚未绑定。
- 动态 context join：`research/pg348_dynamic_context_dataset_v1.json`（2080 行，train=960、implementation_holdout=1120，training eligible=0）把动态 response/role/reset token 接入七轴 context；dataset SHA=`d1d21548359d7a2b6bc06d06824f7a5f48cd11ff86c069d2b0b22da61127c2fa`，vocabulary SHA=`1a012f3b4effbb0fe772713e1b94c4d7b867467f57ba449c990be71715db31a2`，information audit SHA=`782d7f74963c322c9c4e5e2529fad55657efe4d362f1df4b8e5e8ae840dd0a10`；failure 轴仍未观察，不能伪装成已学会排错。
- 周末远程 A800 GPU0 只做 context-only representation candidate（target_tokens_read=false，不读/复制 payload、response body、oracle/evaluator sidecar）：报告=`research/pg348_dynamic_context_a800_representation_v1.json`，文件 SHA=`6669c205ac3432a2c5fb168f4a17a870da8b832ebe154b143c536d0e6417b6f6`；checkpoint SHA=`a5307ca09516be6d9744b652d4ed6925ac82a18e4849b1105155270960ac0a30`；独立 audit=`research/pg348_dynamic_context_a800_representation_audit_v1.json`，SHA=`ab028069b25200df317ed711d538a31bb1fdc928c94303a15f90a228d5886bf1`，状态=`passed_representation_observation_blocked_capability`，失败原因为 `information_gate_not_passed`。三 seed train loss=`6.123006/6.131076/6.139293`，holdout loss=`6.126475/6.133913/6.152247`，holdout predictive entropy=`6.124885/6.128342/6.126525`，unknown token=0；promotion、memory、payload、vulnerability 全部 false。
- 该轮只证明动态 token 能进入表征候选，不能证明模型会发 probe、会生成 payload、会发现漏洞。下一动作是给动态 lane 接入 evaluator-only 的 typed effect、candidate/reference/negative、failure→repair、belief/replay 与 role-bound evidence SHA；在这些字段和审计通过前，动态 rows 仍是 diagnostic/ASK，不能进入 capability 训练或长期记忆。
- 本轮动态专项回归：`python -m pytest -q tests/test_pg348_dynamic_runtime.py tests/test_pg348_dynamic_shapes.py tests/test_pg348_dynamic_context_dataset.py tests/test_pg348_dynamic_representation.py tests/test_pg348_fixture_registry.py tests/test_pg348_surface_projection.py tests/test_pg343_a800_target_conditioned_smoke.py tests/test_pg347_multi_impl_slot_audit.py`=`30 passed`。远程训练结束后 GPU0 回到 1 MiB/0%/无 compute app，本地 `docker ps` 为空。当前 `research/improvement_rules.json` 文件 SHA（外部核对）=`488642bac0928e3f48ee7cad3a3407371418b2602d58b755a983a676e49bdb94`。

### 2026-08-09：PG-348 动态 payload-shape 词表与 A800 候选（扩容后仍 blocked）

- 用户要求把“payload 词表”纳入研究。采用 source-grounded 的抽象 `payload_shape_ref` Rule-IR slot，而不是把可直接执行的原始 payload 字符串塞进模型。新增 `research/pg348_payload_shape_ontology_v1.json`（文件 SHA=`77ca01ed50e7070baa0fb69c99c8c777b6f6b317b7a031973bb97b222e206495`），来源包括 [OWASP WSTG reflected XSS](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/01-Testing_for_Reflected_Cross_Site_Scripting)、[OWASP WSTG SQL injection](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/05-Testing_for_SQL_Injection)、[OWASP WSTG XML injection/XXE](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/07-Testing_for_XML_Injection) 和 [PortSwigger XSS contexts](https://portswigger.net/web-security/cross-site-scripting/contexts)。来源只用于定义 HTML/属性/DOM/脚本、SQL 字符串/数值、XML 文本/属性、查询/路径/JSON/表单等形状与编码维度；原始 probe、响应、oracle 和可执行字符串仍只在 evaluator sidecar。
- 动态采集器 `scripts/collect_pg348_dynamic_typed_rows.py`（SHA=`d06456d832c2ab722510b636990c01fd20a2f20dbde3c750ccc513c23486414b`）完成 520 route × candidate/reference/negative/replay/failure repair：dataset=`research/pg348_dynamic_typed_source_rows_v2.json`，2600 rows，train=1200、implementation_holdout=1400、18 implementations/source IDs、typed/fresh/negative/replay=2600/2600、failure repair=520，文件 SHA=`ab3ee1a967d257b56bc82fe3dbf1cf0cd20b5f6a851c2bedd68ea935a56823b8`。信息审计 status=`diagnostic`、failures=`[]`、context firewall=0、vocab coverage=100%；仍不把这些 rows 直接晋级为 gold，promotion/training/memory/payload/vulnerability 全关。
- 词表已由旧的约 686/54 规模扩为 `context=1207`、`target=122`、`model_union=1308`、ontology inventory=737；vocab=`research/pg348_dynamic_typed_vocabulary_v6.json`，文件 SHA=`a892694aaa3b4325516a12f7857cdbd748103cbe870f6efb5739020e8865b0d5`。词表构建器 `scripts/build_pg331_web_token_vocabulary.py`（SHA=`5ae073e8431f1c6399cbd6ccf3a66494ac2f7d21680f88919c5776e0c3302dc7`）采用 append-only payload-shape inventory；低频/未知/未观察字段不静默删除。当前观察到 9 个 payload shape token，不足的 SQL/XML/命令/重定向形状只登记为 ontology 候选，必须先有动态 typed rows 才能进入训练。
- 容量已扩大到 `required_context_window=790`、balanced `max_length=1024`；legacy `max_length=72` 明确 truncation fail。模型词表 1308，A800 runner 使用 mini-batch=32；`app/pg295_causal_moe.py` 新增可选 mini-batch 与归一化加权 loss（SHA=`a585e106859a9b023568af9df30b33973ef4e36bc0a4ce3c7b00499afd5a5e79`）。容量审计文件=`research/pg348_dynamic_typed_capacity_audit_v8.json`，SHA=`7ad800fca9af948ee4ed6d2fb3c39927bcedc2362d379d73955b6f2b044f8242`；容量技术上够用，不能用“再加宽窗口”掩盖目标边界学不稳。
- 周末远端 `112.111.7.91:60228` 只使用 `CUDA_VISIBLE_DEVICES=0` 的 A800 GPU0，runner=`scripts/run_pg348_a800_payload_shape_candidate.py`（当前 SHA=`fa0161548db2a15a2584580154e4948f2c7ba83061930c1b87b459e79c8587ef`），无本地训练、无 Docker/网络目标。三次 candidate-only 结果：
  - v1 uniform loss：safe reject=1.0、positive recall=0，模型学成“全部拒绝”；report SHA=`ebdc46948061b207adf5ef43686396ad95e88e2a56581d0244cadc996f4a80e7`。
  - v2 target/context=`8/0.25`：最好 seed positive recall=`0.919048`，但 worst-seed false allow=`523`，不能接受；report SHA=`8ca35762ec3669c988af1d0ca924427995ef7cf2b6901205ab5e3f91e85524d0`。
  - v3 另加 negative-safe=`24`、positive-safe=`3`：seed `34803` false allow=`0` 但 positive recall 仅=`0.022619`，worst-seed 仍 false allow=`523`；report SHA=`636968b89b8f22a0da494e14619e42f711155e05abb7ababcf957f83e0ca9809`。
- 结论：词表与上下文窗口扩容已经生效，但模型尚未学会稳定的 payload-shape/Rule-IR 决策边界；这是可复现的训练失败证据，不是“能发正常 payload”。下一步是配对的正/负决策边界 holdout、受限 Rule-IR 解码和跨实现同形目标；不再靠加 epoch、加学习率或扩大原始 payload 字符串追分。硬门仍要求 worst-seed unsafe allow=0、positive recall≥0.95、ASK/repair/negative 全过，之后才讨论结构容量升级。
- 训练后只读复核：本地 `docker ps` 为空；远端 GPU0=`1 MiB/0%`、无 compute app；GPU1–7 未触碰。当前规则文件外部 SHA=`d9f58809fc44d115eeabd0dab3e7f9b3d5797ab4045cb811ada0ba23157d49c6`；远端历史报告锁定的旧 rules SHA 仍保留，不回写历史报告。

### 2026-08-09：PG-349 source-grounded payload 形状词表与 A800 候选（能力门阻断）

- 用户明确要求 payload 词表。公开资料检索后没有把通用攻击字符串下载进训练集，而是新增 `research/pg349_payload_probe_vocabulary_v1.json`（SHA=`eba5c2aea699f1ef3f01edcacc87b5bacd907dfc100d21d7a255f57edb2beba4`）：113 个命名空间 `probe_shape_*` 预留词元，覆盖 surface context、boundary strategy、syntax category、encoding layer、transport placement、probe variant、oracle kind 和 action class。来源是 OWASP WSTG reflected XSS/SQL/XML 与 PortSwigger XSS contexts；原始例子只允许 evaluator-side 短暂人工复核，模型/训练记录不保存原文。
- 追加器 `scripts/extend_pg349_payload_vocabulary.py` SHA=`c04222ddea0f73f7aea9c940efdca93f31b5a99ff5d1f0f604ab3740c9935ed0`，生成 `research/pg349_dynamic_typed_vocabulary_v8.json`（SHA=`eb6e364a5d25a958e122adf7277eecdea46c3c5fcd43165d82aeee35119e388c`）：context=1372、target=122、model union=1473、payload probe reserved=113，append-only；无 raw payload/response token。
- 这轮先修了真实的信息缺口：页面 visible `data-parameter-role`/`data-encoding-chain` 原先未进入抽象观察，导致旧 v2 context-target 冲突；`app/pg331_loopback_adapter.py` 与 pages_b 生成器已补解析/投影，重采 v5 后 context-target conflict=`0`、decision conflict=`0`。v5 dataset=`research/pg349_dynamic_typed_source_rows_v5.json` SHA=`13aa28c7f6649ef8aba7a1c6a92d13b4b77efad32a55efc4723d0bab7cf3221f`，2600 rows、520 routes、train=1200、implementation holdout=1400、typed positive routes=520、failure rows=520。
- 决策边界只读审计=`research/pg349_decision_boundary_audit_v1.json` SHA=`e12a4ae572943a21631bc5b1c8d2a43b1b92b022bfdacf21a6e7417656815afc`，status=`passed_decision_boundary_diagnostic`、unique context=840、context/target conflict=0、decision conflict=0、train/holdout overlap=0、candidate/reference/negative/replay groups=520。它是可辨识性证据，不是 capability 晋级。
- 最终信息审计=`research/pg349_information_audit_v4.json` SHA=`d84de73536fcfbcc82c419ae561f519e7b8444b84c1eea77d9f4811d4a6f668f`，status=`diagnostic`、failures=`[]`、vocab coverage=100%；容量=`research/pg349_dynamic_typed_capacity_audit_v4.json` SHA=`ab1239cf6493ab8c7fd185984b89288c447fde259c262d51b2fa91085e9d5300`，required window=798、balanced max_length=1024 pass、legacy72 truncation fail。扩词表没有掩盖容量问题。
- 新增 `app/pg349_constrained_rule_ir_decoder.py` SHA=`810b43cc011a338b4175c7e2bd042d772a25cef0d58091531a8707086819ea7e` 与 tests（decoder SHA=`9d619ca84b3a21f5ac7e6fcb97602c07e120603f55a94158fba0cf8fb9c3d328`、vocab tests=`13e15dcf4530ff596520aa4051f6b1d178f3e947d6247c0d4c37aa5503c5f311`、extension=`969ca56e0bae7f82e8b832cb3775df638ffd8852eb4b62604ff3518262ae689b`、wrapper=`00d1dccae5029aada941fd151921fb7d451a6ddd97596340c4ecc730f24a0ddf`）。它只返回抽象 Rule-IR；缺观测→ASK、失败→one-variable repair、非法/原始字段→fail-closed，具体字符串仍由授权 evaluator binder 最后一跳生成。
- 周末远程 A800 GPU0 candidate：`scripts/run_pg349_a800_payload_probe_candidate.py`（运行时 wrapper SHA=`259e72ac6836305f9e35d39c23f1d5886959f5942f7cb3ac6ec3a5b2c4b0e791`）在 `112.111.7.91:60228`、`CUDA_VISIBLE_DEVICES=0`、A800 GPU0 完成 3 seeds=`34801/34802/34803`×1 epoch。报告=`research/pg349_a800_constrained_payload_probe_candidate_report_v1_remote.json` SHA=`0b0678c4d823fd188f25dda609effdb6f813d8426036bb56f76f6e3deb7276db`；内层训练报告 SHA=`68ad5434bf01686c0053504ba294776cc77b0d497b39f3c11e4ee5b5c64f84f8`；checkpoint SHA=`ce120c1d064b62b752a4b2d99b349d623abc6d9b5360d478ade2df97051790df`。训练 gate/决策边界/信息/容量全过，但 inner holdout positive recall 三 seed 均=`0.0`，safe reject=`1.0`，false allow=`0`；结论是“只会安全拒绝”，不是会生成/检测 payload。
- 训练后 decoder 兼容性修正只做了本地 preflight，没有重训 checkpoint：`research/pg349_a800_constrained_payload_probe_preflight_v2.json` SHA=`1c83a2e0d5b1b761d5d3eda75391d113f78e7d1527ba4a6f8088e915e161beca`，abstract rows=2600、constrained safe=1560、forced repair=520、false allow=0、raw output=false。不能把该 preflight 当模型指标；远端旧 wrapper report 保留为不可变快照。
- 规则已登记 `research/improvement_rules.json` 的 `pg349_payload_shape_grounding_vocabulary`，当前规则 SHA=`159ec535ba7ef1489f9d1aa56945caf06621a6f332dbf2052ffc325731f7ebe4`；training/memory/payload/vulnerability promotion 全部 false。历史大权重已由 `research/storage_archive_manifest_v5.json`（SHA=`c72e7fa7503e46cf5fa9ba6b731a137482bb27f7a33d3b9760f6264bd5bfce15`）以 E 盘 archive + D 盘 junction 可逆保存，本轮没有再移动/删除活动 PG-349/PG-348 证据。
- 下一安全动作：把 v5 rows 的 target 扩展为显式 abstract oracle/negative-control presence 与 payload grammar slots，做 source/implementation holdout 的 constrained decoder smoke；随后只在人工确认 binder 后的 disposable loopback 上显示真实 wire。未通过 positive recall、ASK/repair、negative false-allow 和 fresh replay 之前，不得声称“AI 能发正常 payload”，也不得把任何原始字符串写入训练集或长期记忆。

### PG-349 后置复核

- PG-349 相关回归=`51 passed`；全量 `python -m pytest -q --durations=10`=`1422 passed, 1 warning in 184.67s`。本地 `docker ps` 为空；孤立的无脚本 Python stdin 进程已停止；本地没有 CUDA compute app。
- 远端复核：GPU0=`1 MiB/0%/no compute app`，GPU1–7 未查询/未触碰。当前阶段仍是 `completed_candidate_only_blocked`，不是模型能力晋级；规则文件最终 SHA=`159ec535ba7ef1489f9d1aa56945caf06621a6f332dbf2052ffc325731f7ebe4`。

### 2026-08-09：PG-350 抽象槽到真实 wire 的最后一跳

- 用户明确指出：如果最终永远不生成真实字符串，研究就无法让人工审核/复现。结论是分层而不是二选一：模型训练与上下文继续只使用抽象 `transport_ref`、`field_role_ref`、`encoding_ref`、`payload_shape_ref`、`probe_variant_ref`、`oracle_ref`、`negative_control_presence_ref`；授权 evaluator 在最后一跳读取 source-attested template，把一次性 `{{MARKER}}` 绑定成真实值，生成可读 GET/POST wire。真实值可被人工临时查看并复放，但不进入模型、训练集、长期记忆或持久 catalog。
- 新增 `app/pg350_runtime_payload_binder.py`（SHA=`d35383f6f12e383d9e4fd073818d1525767be52ab032ff4a2bff09a56acd049b`）及 `tests/test_pg350_runtime_payload_binder.py`（SHA=`bb4bc5b7b57bbe42b7f437b355014b6fddbcb6d21d2d2388d188a9d730b8709b`）。`bind_runtime_probe()` 纯内存、无网络；仅接受 loopback origin、source/route/field attestation、fresh reset、candidate/reference/negative、replay consistency 和 allowlisted template；`human_review_wire()` 是唯一具体 wire 显示边界，`persisted_projection()` 只保留抽象槽/模板 ID/值与 wire SHA，raw flags 全为 false。stateful lane 额外要求 reset before/after、database clean、teardown。
- PG-331 source-row target 合同新增 append-only `oracle_ref` 与 `negative_control_presence_ref`，旧行仍兼容；缺 evaluator/字段时两槽强制变成 `unknown`、`safe_to_send=false`。`app/pg331_source_row.py` SHA=`62c9d565aa410a28ed35b164aab0c6f924f52d9549b9510311b761a684ab82b4`。
- 新增 `scripts/build_pg350_oracle_slot_dataset.py`，由冻结 PG-349 v5 抽象行构造版本化 target-slot 视图（不复制 raw）。产物 `research/pg350_oracle_slot_source_rows_v1.json` SHA=`d42ca3d1acb2982ea6c485020058ffd8ec62d4ce2ce1bf59ace74b829dc4fc49`：2600 行，train=1200、implementation_holdout=1400，target slot coverage=100%，training_eligible=0，promotion 全 false。新增 `scripts/extend_pg350_oracle_vocabulary.py` 及 `research/pg350_oracle_slot_vocabulary_v1.json` SHA=`069a726fab8d5e77758fb9e48ce3ae0174b3e9104895cad5f00a18586f539c5b`；target vocab=135，append-only、无 raw literal。
- 新增只读审计 `scripts/audit_pg350_oracle_slots.py`（SHA=`6d0832adb5989097800733bd7fbadb6cd269c730b231fc7e0f59798a88570ae5`）和 `research/pg350_oracle_slot_audit_v1.json`（SHA=`e8860fc08e79c51690b0792e49f36ee26af28d16efee716cc27be9ebf23cdd73`）。审计 `diagnostic_only`、failures=[]、context-target conflict=0、vocab missing=0、required window=774、balanced max=1024；`negative_control_presence_ref` 的唯一值产生 zero-entropy warning，predictive-entropy holdout 尚未运行，accepted_training_rows=0。
- PG-350 focused regression=`29 passed`（binder/dataset/vocabulary/audit/source-row/decoder），规则合同=`5 passed`。这轮没有启动 Docker、网络或 A800；PG-349 之前的 A800 候选仍是 holdout positive recall=0、safe reject=1，不能宣称模型已学会 payload。当前 `research/improvement_rules.json` 已登记 `pg350_runtime_payload_binding`，promotion/training/memory/payload/vulnerability 全部 false。
- 下一安全动作：在明确授权的 loopback evaluator 里，用 source-attested template 分别生成一条 GET 与一条 POST 的 candidate/reference/negative/fresh replay，人工查看 ephemeral wire 和 typed evidence；通过后再做 PG-350 target-conditioned A800 short smoke。不能把 binder 正确当成神经模型生成能力，也不能把具体 wire 回灌到训练上下文。

### 2026-08-09：PG-350 真实 wire 最后一跳回放（evaluator-only）

- 用户确认“最终仍需真实原始字符串”。已实现 `scripts/run_pg350_runtime_binding_replay.py`：模型/Rule-IR 只提供 transport、参数角色、编码链、payload shape、oracle、variant 等抽象槽；source-attested evaluator template 在内存中绑定 `{{MARKER}}`，通过本机 `127.0.0.1` 动态 runtime 实际发送一次 GET 与一次 POST。`human_review_wire()` 只在终端显式显示，不能进入 JSON、模型上下文、训练集或长期记忆。
- 回放范围：3 seeds=`35001,35002,35003` × 2 routes（GET=1、POST=1）× candidate/reference/negative/replay；每角色 fresh reset before/after。另有 `unsupported_variant → source_attested_candidate` 的失败→修复动作变化。
- 结果：6/6 candidate typed、6/6 reference typed、6/6 negative clean、6/6 replay consistent、6/6 failure action-change；audit=`passed_evaluator_only`，raw firewall violations=`0`。这证明 evaluator binder/wire/typed oracle 闭环，不证明神经模型已经能生成 payload，也不证明任意网址存在漏洞。
- 产物：runner SHA=`c52ec78d3052f4f988a7425bb0b40ce2eeebb152a3f65ab479143339fc0e0fbf`；audit script SHA=`7ee9f0e11722b3e20f4db5fe8f6af5d987a97075084af16971074ff586a0470f`；report=`research/pg350_runtime_binding_replay_report_v1.json` SHA=`281528c0d3a59200120c6d248a6215c0d0d8b180849e02ba0de973e310eab34d`；sidecars SHA=`f46084651cad32b925b98e19d305a1cbacd6b320b8eca68bf5a595f5bbc00ed5`；audit SHA=`b2c273ead860d5e39ed6e3b33cbeb6b0fdddeacd26a1c78b850f5cb035e86dbf`。
- 该 runtime 是 PG-348 synthetic dynamic loopback，`implementation_independence=false`；promotion/training/memory/payload/vulnerability 全部 false。下一步是把相同 binder/sidecar 合同接到第二个独立、已授权的动态实现，并先复做信息熵/神经正例召回门，再谈 A800 capability smoke。
- 最终回归：`python -m pytest -q --durations=10`=`1438 passed, 1 warning in 188.75s`；warning 仍是既有 Torch nested-tensor。当前本地 `docker ps` 为空，本轮未启动 A800。

### 2026-08-09：PG-351 抽象组合候选与 raw wire 边界明确

- 用户明确指出最终复现仍必须得到真实原始字符串。项目规则因此明确为“两层”：模型只预测 `transport_ref`、`field_role_ref`、`encoding_ref`、`payload_shape_ref`、`probe_variant_ref`、`oracle_ref`、`negative_control_presence_ref` 和 `safe_to_send`；授权 evaluator 最后一跳读取 source-attested 单标记模板，在内存中绑定一次性 canary 值，生成真实 GET/POST wire，供人工临时查看、发送和 typed oracle 复放。raw wire 不进入模型上下文、训练记录、长期记忆或持久 catalog，只保存抽象槽、模板 ID、来源和 SHA-256。缺观测、缺 reset、缺 typed evidence 或阴性对照时不绑定，模型目标必须是 `ASK/safe_to_send=false`。
- 新增规则 `pg350_runtime_wire_generation`，把上述 evaluator-side raw 生命周期、GET/POST、candidate/reference/negative/replay、fresh reset、evidence hash、stateful disposable 例外和禁止公网/外联/凭据/时序通道写成硬合同；规则文件最新 SHA=`890046847e2f8ae38d4e18da3b8120f06423fa0177956de869d33edc96a20661`。
- PG-351 组合数据集 `research/pg351_ask_oracle_composition_dataset_v1.json`（1832 条：train=1152、implementation_holdout=680）把缺 typed 的 ASK 行与 PG-350 抽象 oracle-slot 行合并，精确去重；audit=`diagnostic_candidate_only`，`training_eligible=0`，raw/context firewall=0，promotion 全关。dataset SHA=`dc06bed33651e71266ce002d94c59e94be5f9ace61ddfcbd505981419c74cbfb`，audit SHA=`145ed66c24f80848c7cbcfe287b141e2fc2f473bd512ffd49cbaff93f9afd1f0`。
- 周末授权远程 A800 GPU0 做了 3 seed×1 epoch 的 target-conditioned candidate smoke（runner SHA=`d650f1ea175197da1fa73396db267257c4885fc27adc9e3ca6dffb990cf25e09`，`CUDA_VISIBLE_DEVICES=0`）。报告=`research/pg351_a800_ask_oracle_composition_candidate_v1_remote.json` SHA=`3300a8f67ab7ef6c0b512c28e9380eb9a5e084da0ae1f489698a2045bc748027`，checkpoint SHA=`c61b754820084edf4aa8dc725c109d8834e4107bbdb3af6916d22071750a7fa3`；A800 结束后 GPU0=`1 MiB/0%`、无 compute app。
- 结果必须按能力拆开看：族外 ASK 最坏=`0.952381`、负例误放=`0`、最大 holdout entropy drop=`0.008113`；repair=`0`、abstain=`0`、positive-action=`0`，positive recall 最坏=`0.705882`。结论是模型开始学会“先问”，还不会稳定“修复并选择可发送抽象槽”，不能宣称会生成/检测可迁移 payload；candidate、memory、payload catalog、vulnerability promotion 全部关闭。
- 下一步不是把原始字符串塞进训练集，而是补 repair/abstain/positive-action 的跨实现 hard-negative 与 failure transition，再让模型输出抽象槽，由 PG-350 binder 在第二个独立授权实现上生成 ephemeral wire；只有 fresh typed replay 和 worst-seed 正例/阴性门同时通过，才可展示“模型选槽→binder 生成真实 wire→oracle 验证”的完整过程。

### 2026-08-09：raw payload 的“最后一跳”不是禁止生成

- 用户澄清最终交付必须能看到并复现真实原始字符串。规则解释固定为两层：`model_output=abstract Rule-IR slots`，`system_output=ephemeral concrete raw value + GET/POST wire`。因此 raw 不是在系统任何位置都禁止，而是禁止进入模型上下文、训练记录、长期记忆和持久 catalog；授权 loopback evaluator 可以在最后一跳从 source-attested 单标记模板生成一次性真实值，供人工审核、发送和 typed oracle。
- 已把该语义写入 `research/improvement_rules.json` 的 `pg350_runtime_wire_generation.final_output_contract`。binder `app/pg350_runtime_payload_binder.py` 的 `human_review_wire()` 是唯一具体 wire 显示边界，`persisted_projection()` 只保留抽象槽、模板 ID、来源与 SHA-256。
- 2026-08-09 的本地 synthetic dynamic loopback 复放（临时 seed=35099，仅写入系统临时目录）实际显示了 GET 与 POST：GET 的 query 值由运行时 canary 加单引号并按 URL 编码，POST 的 form body 同样由运行时 canary 加单引号并按 form 编码；2/2 candidate、2/2 reference、2/2 negative clean、2/2 replay、2/2 failure-action-change。binder/replay focused tests=`8 passed`。
- 这些 wire 是非破坏性 evaluator canary，不等于神经模型已经学会任意 XSS/SQL 原始攻击字符串；当前 PG-349/PG-351/PG-356 候选的正例/repair/abstain 仍未过最坏种子门。以后如果要在授权 Pikachu/DVWA 等本地靶场复现，仍须由 evaluator 侧绑定已审阅模板、fresh reset、GET/POST、candidate/reference/negative、typed oracle 和证据哈希，模型不能自行注入任意 literal。

### 2026-08-09：PG-359/PG-360 信息索引与 slotwise 复核

- PG-359 修正了 context index 的枚举缺口：`belief_process_step=failure` 与 `failure_failure_class=blocked_variant` 必须是显式词元，不能落入 unknown。校正后的 dataset=`research/pg359_context_index_dataset_v1.json`（SHA=`88f47f3198eed9d8dc3b449b9e2f07402c46cb3cb19a99ee4a8f7cef3e73cd00`），audit=`research/pg359_context_index_audit_v1.json`（SHA=`7a28f09a116a7a3fac44b68e77751287cd27548c3dba10e71a4b570ef4f700e2`），builder SHA=`ecae3d30c8b0c5994aec8f80f43127890fca0da3863b83bcfcc3ea8edc54add5`；1832 rows，原 context 全保留，未读 target/evaluator sidecar，raw=0。修正前 E1/E2/E3 只能标 stale diagnostic，不能作为能力证据。
- PG-360 使用校正后的 PG-359 context 建立 12 个独立 Rule-IR slot query（21984 slot rows，max_length=627），dataset SHA=`abb94768abdc00d3dcf2ec5874dd3ab23dffe71db6aa2d55f65b4cde43b767c1`，audit SHA=`e9cb371b275e39e82931dfaf65427e8360beeb0f411bd5be37dc0de31f34f427`，runner SHA=`de10073705143ca9ca4a5a464f9f32fbd45818a2a2ac3869657d6b06c9d91b64`。校正索引后的 A800 GPU0 E3 candidate report=`research/pg360_a800_slotwise_e3_corrected_index.json`（report SHA=`633d0c8467de77153b380d5ed56fcc092780718c287ba0c00bf8fcd6414c6d26`，checkpoint SHA=`5e2eb0a8c94d3328e8ffb5b05f8f103db8dd474ec651199e4d9851c478f7d2c4`）：entropy drop 最坏=`0.030507`（过 25% 门），ASK=`0.952381`，但 repair=`0`、abstain=`0`、positive-action=`0`、Rule-IR assembly=`0.705882`，negative false allow 最多=`68`；因此仍是 candidate-only blocked，不是可发 payload 的模型。E2 的 balanced 版本 entropy drop=`0.41916`，明确不能用过度平衡换分数。
- payload 词表现在明确拆成 `surface_context`、`parameter_role`、`encoding_chain`、`syntax_category` 和 `validation_oracle` 五个可组合轴；规则已把新候选的 `syntax_category_ref` 设为必需槽，旧候选缺槽只能 diagnostic。模型仍只输出抽象槽，最后一跳 evaluator 才从 source-attested 模板生成一次性真实 GET/POST wire；真实 wire 可人工临时查看，但不进 context、训练集、长期记忆或 persistent catalog。规则文件本轮 SHA=`7602beeb749f4a654790b863c147834a7df0c33b5609f6a9bd1de3cecf9ef103`。
- 本轮没有启动 Docker/公网目标；远端 A800 GPU0 已回到 `1 MiB/0%/无 compute app`，GPU1–7 未触碰。下一动作应是为 `syntax_category_ref` 做独立 slotwise 数据/holdout，再接 PG-350 binder 的第二个独立授权实现；不能用本 synthetic binder 的成功替代神经正例召回。

### 2026-08-09：PG-360 采样、容量与 fail-closed guard 对照

- E4 平方根重采样（不改变 split/context，只把稀有抽象动作的训练出现次数调到 `sqrt(count×max_count)`）报告=`research/pg360_a800_slotwise_e4_sqrt_balance.json`，SHA=`4c2ecb4bd4ef5a8ca5ab4b8c7b9aac7391bb01cf03f439aac6871c4bc9c3a5d6`，checkpoint SHA=`82f2dfd0613b885cdfb5c435a93709727f2276eca14a640fd04d3a2ce52174f6`。positive-action 最坏升到=`0.666667`，但 predictive entropy drop=`0.489658`、negative false allow=`68`、repair=`0`，因此证明“重采样不是根因”，candidate blocked。
- E5 容量对照（`d_model=256,n_layers=4,experts=4,expert_hidden=512`，经验分布不变）报告=`research/pg360_a800_slotwise_e5_capacity.json`，SHA=`fc0a5c8251aa67c66187433668735621b812ecbf4d885e72bb4a5794cac34dd1`，checkpoint SHA=`a8534ebf584ba97068d108475779f64ba43b415cb0580cdbfc6f8c5801e75286`。positive recall=`1.0` 但 entropy drop=`0.732751`、negative false allow=`68`、repair=`0`，容量增加反而过拟合/熵塌缩；不能把“模型不够大”当作当前主因。
- E6 把现有 `app/pg349_constrained_rule_ir_decoder.py` 接到 slotwise 解码后做 3-seed A800 candidate-only：报告=`research/pg360_a800_slotwise_e6_guard.json`，SHA=`6261d16b6415aaefa798e206e7ef2d5b359905e7258f74b79a33d4797f1220ae`，checkpoint SHA=`c4eabdb8492d64c4f914093b7d092c2fa655a187d9070530f8f3337f2f409130`。裸模型仍 repair=`0`、positive-action=`0`、negative false allow 最多=`34`；context-only guard 后 ASK=`1.0`、repair=`1.0`、negative false allow=`0`，但 positive-action/positive/assembly 仍=`0`。这是安全层效果，不是模型能力；原因是当前 source rows 的完整 typed/failure/negative 条件不足，guard 正确地全部阻断发送。
- runner 现在记录裸模型与 guarded 指标，支持可选 `--apply-rule-ir-guard`、`--sqrt-balance-slot-values` 和容量参数；当前 SHA=`28af18be2fcb1cb054b6ba19720a7241bd00ee5d65ee703062de0e54b398af71`。相关本地回归=`10 passed`（PG-360/PG-349 decoder）。规则文件本轮 SHA=`f4f62147ac357084abf9a47c3406e3529b2240b3b73f65e32a1895cf8f40086a`。
- 结论与下一动作：不再盲目堆容量/采样；必须补完整 failure→repair、typed candidate/reference/negative、fresh reset、evidence hash 和显式 `syntax_category_ref` 的 source rows，再训练。当前 guard 证明模型不能在缺证据时发送；要得到正例，先补数据而不是放宽 guard。E6 结束后远端 GPU0=`1 MiB/0%/无 compute app`，GPU1–7 未触碰。

### 2026-08-09：C/D 存储审计

- 当前可用空间只读核对：C=`66.21 GiB`、D=`159.11 GiB`、E=`152.71 GiB`、F=`89.80 GiB`。D:\workspace\blackboxanalyze\artifacts 中的历史大权重已经是指向 E 盘 archive 的 junction，D 盘没有重复实体；活动 PG-331/PG-348/PG-359/PG-360 证据未移动。
- 项目内唯一明显可再生大文件是 Next.js Turbopack cache `frontend/.next/dev/cache/turbopack/v16.2.12`，约 `290,339,162` bytes；没有 node 进程。删除命令被当前执行策略 fail-closed 拒绝，因此本轮没有删除任何文件，也没有把研究数据误删；该候选已登记到 `research/improvement_rules.json.storage_governance.last_storage_audit`，后续需在明确可执行的可逆清理窗口处理。

### 2026-08-09：PG-361 payload 五轴槽位与真实 wire 最后一跳

- 用户明确要求：抽象词表不能成为“永远不输出真实 probe”的借口，最终必须能在授权本地靶场生成可读、可复现的原始 GET/POST。规则固定为两层：模型输出 `transport_ref`、`field_role_ref`、`encoding_ref`、`syntax_category_ref`、`payload_shape_ref`、`probe_variant_ref`、`oracle_ref`、`negative_control_presence_ref` 与 `safe_to_send`；evaluator 依据 source-attested 模板在内存中把一次性 canary 绑定成 concrete raw value/wire，人工可临时查看，发送后只保存投影与 SHA-256。
- 新增 `app/pg361_payload_shape_slots.py`（SHA=`bd0a4defa14e401af9b76327f7317ad51d659c8d62d0ae8f9697cd5fecef4d71`）与 `scripts/build_pg361_payload_shape_slot_dataset.py`（SHA=`9d6c275f537356443fedb8e7185aaf02c90bb826965f5c8f2a39ec160dffaf4d`）。`syntax_category_ref` 是新候选的必需抽象槽；允许的 grammar class 为 marker、delimiter_boundary、structured_value、expression_node、boolean_branch、parser_node、state_transition、redirect_control。原始 payload、URL、response body、route/evaluator literal 会被拒绝。
- `app/pg331_source_row.py` 增加 append-only `syntax_category_ref`（SHA=`a7be5d3d4475554c6f9c6fc8e669e8f9abeb56cb6ec9d777fadedcdfd2d59fc6`）；缺槽的旧行仍可读取，但新候选只能 diagnostic/ASK。`app/pg350_runtime_payload_binder.py` 增加 syntax 类别与 template 类别一致性校验（SHA=`c8ab79f756d438391f703c81c929d34caebab2d9687c4bd48aa4a6bd9259cf48`）；它仍只在 loopback、fresh reset、candidate/reference/negative/replay、typed evidence 通过时绑定真实 wire。
- 由冻结 PG-350 evaluator-side 行和 PG-348 registry 做只读转换，产物 `research/pg361_payload_shape_slot_source_rows_v1.json`（SHA=`c6b19859c1d79227a9244a9ca43281c909934662834dfa735451f51c3af207bd`）：2600 rows，train=1200、implementation_holdout=1400，8 个 syntax 类别，training_eligible=0、promotion 全部 false。它是 target-slot diagnostic view，不是新 gold 数据，也没有接触靶场。
- 本轮 focused 回归=`27 passed`，PG-331/349/350/361 交叉回归=`60 passed`，全量回归=`1472 passed, 1 warning in 331.24s`，py_compile 通过。另在临时 loopback synthetic runtime 做了 1 seed×GET/POST、candidate/reference/negative/replay 的真实 wire 最后一跳：`confirmed_positive=2/2`、negative/replay/failure-action-change 全通过；终端显示的 wire 只在进程内出现，report SHA=`6163f7d3e2ca55bb5e88d4827f326cb93220bfbeb5782636a7013a7b83bdd179`、sidecars SHA=`2c8c3be043acc1f648ad7744ac6dd7b01ff0580f510afad677e64c93533659ba`，没有写入 workspace、模型或长期记忆。该结果只证明 binder/evaluator 闭环，不证明神经模型已经会自主生成任意漏洞 payload。
- 新增只读审计 `scripts/audit_pg361_payload_shape_slots.py`（SHA=`ae3fc76fc93f048157a83e5445509e5838d7e16f51025c0dacaa7db484080023`），报告 `research/pg361_payload_shape_slot_audit_v1.json`（SHA=`436840961ab9fed8f9f644e5a44db562569cccbd7ee4bce74f6134ed41d29d03`）：invalid=0、raw_hits=0、context-target conflict=0、train/holdout overlap=0、training_eligible=0；`predictive_entropy_not_run`，所以状态仍是 diagnostic_only。
- 规则文件已登记 `pg361_payload_shape_syntax_slot_contract`，当前 SHA=`84c18f0a58d1d69a13aa983ea7a1acad9342efbcbf93f85a7da8e2ea068cdd29`。下一动作：把同一五轴槽位、失败修复、阴性对照和 binder 合同接到第二个独立授权动态实现；在神经 positive-action/repair/negative/fresh replay 最坏种子门通过前，不启动新的 A800 capability 训练、不晋级 payload catalog。

### 2026-08-09：PG-361 动态 typed rows、slotwise 数据与 A800 候选结果

- 520 条 PG-348 动态 route 通过 loopback-only collector 做了 fresh candidate/reference/negative/replay/failure 采集。`research/pg361_dynamic_syntax_typed_source_rows_v1.json` SHA=`6f3d07e34dc557e9b685b0e35eb4075d75c59d376ee43054b66dcfdca2ec14b5`，sidecar SHA=`7caf2ce2d1e20b1366e4abbe81be60b19d9a52f7dba989849927f545328a5350`，report SHA=`1956730f3ed57f7581d2c79d34cc7d3916d75960f5eb73597b750ac591f5fdf1`，信息审计 SHA=`cb1770835524e47340fbb0c963002ce855860a0b10ccac2ee677985de49d38c3`。总计 2600 source rows（train=1200、implementation_holdout=1400，GET=1550、POST=1050、failure=520、syntax 类别=8）；raw/context firewall=0，training_eligible=0，predictive entropy 尚未运行，仍是 diagnostic-only。
- `scripts/build_pg361_slotwise_dataset.py`（SHA=`889161b767b5bda523d5ab7bb7fe28a2f2b555f9ace3b1a9f037b60f7413b745`）把每条 source row 展开为 13 个抽象 slot query，得到 `research/pg361_syntax_slotwise_dataset_v1.json`（33800 条、SHA=`43d3d87628803bfd755e6bba3dca9f290155b7fb234fb2dac422da983892bfe8`）；slotwise audit=`research/pg361_syntax_slotwise_audit_v1.json`（SHA=`85490de2c8a9d2e165e32e1f623dd6447623e8add0773c2b527a3d3c04db40b8`），13/13 slot、词表缺失=0、raw=0、promotion/training 全关。数据量增加没有改变信息/实现留出硬门。
- 周末按规则在远端 `112.111.7.91:60228` 仅使用 `CUDA_VISIBLE_DEVICES=0` 的 `NVIDIA A800-SXM4-80GB` 做了 3 seed×1 epoch、`max_length=612` 的 candidate-only smoke。runner SHA=`1a8ded026cb1a3a4fb99c256feef71ae7580909217becfd80894c2b65d7835f6`；report=`research/pg361_a800_syntax_slot_candidate_v1.json` 文件 SHA=`faf47ff1374de179362ea50779edaa45c1a16c227ee3cbab661586cd5826aefe`、内部报告 SHA=`7fb1470b7ee324b0f70a2c42cbce85d2f78e30c2ecacfedc8f7f0905b0ccfff4`；checkpoint SHA=`8c53eeb5dfc1ed7caa8d74fd7a19010f24bf8aea23f421d7a41bc2fc46386a80`。训练完成后 GPU0 回到 1 MiB/0%，GPU1–7 未触碰。
- 结果：窗口/模型技术上可运行，worst-seed predictive entropy drop=`0.016442`（熵门通过），但 ASK=`0`、repair=`0`、abstain=`0`、Rule-IR assembly=`0`、positive-action=`0.333333`、negative false allow=`560/560`；因此明确 `blocked_candidate_only`。它证明“扩容后仍然不会安全决策”，不是 payload 能力，也不允许把 checkpoint/rows 晋级训练集、长期记忆、payload catalog 或漏洞声明。
- raw wire 语义再次固定：模型只输出抽象槽；授权 evaluator 最后一跳才按 source-attested template 绑定一次性 marker，生成真正的 GET/POST 原始字符串供人工查看、发送和 typed oracle 复放。raw 字符串不进模型上下文/训练/长期记忆；只有 wire 的方法、参数位置、编码链、模板来源、请求哈希、响应投影和证据 SHA-256 可持久化。当前还没有“神经模型选槽→独立实现→typed fresh replay”通过记录。
- 当前规则文件在本轮补充 dynamic collection、slotwise dataset、A800 failure 和 raw last-hop contract 后 SHA=`7fe4ad92da9cfc74a5769e68feecc7fc17d6b79f1188d4dabc2a75672184ff3a`。下一安全动作是补第二个独立动态实现及真正的 failure→repair/negative hard-negative，再训练；不能用更大窗口、更大词表或更大模型掩盖 `ASK/repair/negative` 全部失败。

### 2026-08-09：PG-361 slotwise 评估前缀错位修复与第二轮 A800 结果

- 复核 PG-361 v1 后发现评估器 bug：训练样本的条件是 `context + [SLOT_QUERY_BOS] + slot_query=<slot> + [SLOT_QUERY_EOS] + [TARGET_BOS]`，但 `_slot_prediction()` 漏掉了 schema query，直接把 `[TARGET_BOS]` 接在页面上下文后。v1 report 因此只能标为 `stale_evaluation_protocol_diagnostic`，不能解释成模型能力；该修复由 `scripts/run_pg360_a800_slotwise_candidate.py` SHA=`52a530647086641dfc3fa45fcb592fa5855b5118f2744d624ba86f9b95c5b286` 固化，回归测试=`6 passed`。
- 用修正后的同一 dataset/audit、同一模型容量、同一三 seed 在周末远端 A800 GPU0 重跑 candidate-only。v2 report=`research/pg361_a800_syntax_slot_candidate_v2.json` 文件 SHA=`2c30c85976c2fedfd86852995980ffaaeac5f0bb26126f01d6ef2f4376c5ae59`，内部 SHA=`2a8452f56888f037132f81776788b371e073f7282f09b4b166947ebe73ed9d72`，checkpoint SHA=`f49fd126ebb22a5bd6b00ff2813c393c4a5a315efd437ccf91c7d38cbb863689`；训练后 GPU0=`1 MiB/0%`，GPU1–7 未触碰。
- 修正协议后结果仍是 `blocked_candidate_only`：worst-seed entropy drop=`0.016442`，ASK=`0`、repair=`0`、abstain=`0`、Rule-IR assembly=`0`、positive-action=`0.322619`、positive recall=`0.359524`、negative false allow=`560`。第三 seed 负对照误允许仍为 `202/560`。因此 v1 的负面对照全误允许部分来自评估错位，但 v2 证明核心能力边界仍未学会；不能晋级数据、记忆、payload catalog 或漏洞声明。
- 当前规则文件已登记“v1 stale evaluation + v2 corrected evaluation”，最新 SHA=`0a349e1ca6711867eaf1d5d3ba54c7134ea1a05a71dab1b92dd317c3ea6b5ff7`。下一动作仍是显式 hard-negative/ASK/failure→repair 轨道与第二独立动态实现；不再仅扩大窗口/词表/容量。

### 2026-08-09：PG-362 完整 Rule-IR 目标与 raw wire 最后一跳复核

- 发现 PG-361 动态 source rows 的原始 target 只有 12 个显式槽，缺少 PG-351/PG-360 合同中的 `ask_reason`。新增 `scripts/build_pg362_full_rule_ir_dataset.py`，只从已有抽象 `question=ask_*`/action 枚举推导有限的 `ask_reason`（`ask_failure→failure_feedback`，其他 ASK→`typed_evidence`），再按固定 13 槽顺序重建完整 target；不读取 evaluator sidecar、原始请求/响应或 payload。builder SHA=`7841e5b8212d2a172b1b7acfcf1cf93a3efdc23dabe6a6e76a11d9dd0d57ed2a`。
- PG-362 dataset=`research/pg362_full_rule_ir_dataset_v1.json`，SHA=`2f2e270a9143e9488a7e9206cace944abb3b0f37b31bd8b3059bf2bc3c3f4d35`；2600 rows（train=1200、implementation_holdout=1400）、13 target slots、265 unique target sequences、最长 context+target=621。audit=`research/pg362_full_rule_ir_audit_v1.json`，SHA=`2d1d76179575d0b5de80442a05b094def02a3d02e76f5f38cc719b0a7acfb081`，audit script SHA=`f2881fe314e2b8dea0a2cb12e81e19314a7e04125e48dc3dfdccb1ebd54c8ed9`；raw hits=0、vocab missing=0、结构 status=`diagnostic_candidate_only`，但 `training_eligible=0`、predictive entropy 尚未运行。
- 周末远端 `112.111.7.91:60228` 的 A800-SXM4-80GB GPU0 完成 PG-362 3-seed×1-epoch candidate-only（显式 `CUDA_VISIBLE_DEVICES=0`，GPU1–7 未触碰）。runner SHA=`08b9da56af1e9eafd017abd9d309e2a7d35258c750572bbac600f263a5140e07`；report=`research/pg362_a800_full_rule_ir_candidate_v1.json` 文件 SHA=`b8615aaeaa0ed1a15798eb67420f34b158cf4e3e993cb5aad9961bd8041a5c9`、内部 SHA=`75900c661ba97c1b44f3af1326ba116190977c34885361aac07e738375416eb2`；checkpoint SHA=`de90955ea4b435c5f077eb8500a4a80fae91a4632afc75c3291db634c4180cb9`。
- PG-362 结果：熵相对下降最坏=`0.001360`（信息熵门通过），但 implementation holdout ASK=`0`、repair=`0`、abstain=`0`、positive-action 最坏=`0.563095`、positive recall 最坏=`0`、negative false allow 最坏=`560/560`、完整 Rule-IR sequence exact=`0`。结论是跨实现条件泛化仍失败，候选、训练晋级、长期记忆、payload catalog、漏洞声明全部关闭；不能把完整 target 契约通过误报为模型会发 payload。
- raw wire 规则进一步澄清：最后确实必须产生真实原始 GET/POST 才能复放。模型只输出抽象五轴/Rule-IR 槽和 variant；`app/pg350_runtime_payload_binder.py` 在 evaluator 最后一跳读取 source-attested 单一模板，将可审阅的具体探针/非破坏性 exploit fixture 与一次性 runtime canary 在内存中绑定，`human_review_wire()` 临时显示并可发送到已授权 loopback；`persisted_projection()` 只留下槽、template/source/request hash、响应投影和 evidence SHA-256。原始字符串因此“会生成”，但不进入模型上下文、训练数据、长期记忆或持久 catalog，也不能由模型任意提供 literal。缺 scope、fresh reset、candidate/reference/negative、typed oracle 或 replay 时仍强制 ASK/abstain。
- PG-362 相关 focused tests=`10 passed`（full dataset/audit + corrected slotwise candidate）；本轮 A800 结束后 GPU0 应回到 1 MiB/0%，本地无靶场容器。当前 `research/improvement_rules.json` 外部 SHA=`d369e4e8c5a01176fe7de9c419e5f53f87a0fac9c447cf27fa5232e1503ba934`。下一动作是第二独立动态实现的 hard-negative/ASK/failure→repair 与 binder fresh replay，而不是继续把 raw 字符串塞进模型或用 guard 分数冒充神经能力。

### 2026-08-09：PG-363 全上下文 pooled Rule-IR 候选与真实 wire 边界

- PG-362 的主要架构假设是 structured head 只读最后一个上下文 token；PG-363 新增 `app/pg363_pooled_rule_ir.py`，对全部有效 context hidden states 做 learned-attention + masked-mean + last-boundary pooling，再用独立 slot query 联合预测完整 13 槽 Rule-IR。模型仍只读抽象 context，target 只作为 label，不读取 evaluator sidecar、raw 请求或 raw 响应。
- 本地回归：`tests/test_pg363_pooled_rule_ir.py`=`2 passed`；decoder SHA=`a1543741ac500df7b9267c2bcebac92e9f9d2edfc24847e5f927aa1dd2d7a49a`，runner SHA=`6c3620b8be3f98a5f0dbc6e7f95e0b4e4fbd103d22bcd0c2cea9bddc743546d`。
- 周末授权远程 `112.111.7.91:60228` 仅用 `CUDA_VISIBLE_DEVICES=0` 的 `NVIDIA A800-SXM4-80GB GPU0` 完成 3 seed（36301/36302/36303）×16 epochs candidate-only；训练后 GPU0=`1 MiB/0%`、无 compute app，GPU1–7 未触碰。输入数据锁定 PG-362 dataset/audit，required context window=`621`，没有沿用 legacy 72。
- 产物：report=`research/pg363_a800_pooled_rule_ir_candidate_v1.json`，文件 SHA=`a77e8b6c65e8d1330b8815917d0605ec6935d17749492c130ab7d0b3696fbde4`，内部 report SHA=`bda4050143872a95efb61f7abca952bfd8cc295a64df7bbd787bbe40b1692ee0`；checkpoint=`artifacts/pg363-pooled-rule-ir/pg363_a800_pooled_rule_ir_candidate_v1.pt`，SHA=`09ab8e7c22252791cdfcaa26a853d4c88650eab59f95f37c9eed7e84410235bb`。
- 最坏种子：relative entropy drop=`0.002016`（信息门通过），ASK=`0`、repair=`0`、abstain=`0`、positive-action=`0.666667`、negative false allow=`560`、Rule-IR sequence exact=`0`。结论为 `blocked_candidate_only`：pooling 修复了读取位置，不等于模型已学会安全决策或可迁移 payload；训练/记忆/payload catalog/vulnerability promotion 全部关闭。
- raw payload 的最终交付不是被取消，而是严格分层：模型输出五轴/Rule-IR 抽象槽和 allowlisted variant；授权 evaluator 最后一跳从 source-attested 模板绑定一次性 canary，生成真实可读 GET/POST wire，人工可临时审核、发送、接收 typed oracle 并复放。raw 不进入 context、训练记录、长期记忆或持久 catalog，只留模板/来源/请求哈希/响应投影/evidence SHA-256。没有 scope、fresh reset、candidate/reference/negative、typed oracle 或 replay 时仍强制 ASK/abstain。
- 规则文件已在 `pg361_payload_shape_syntax_slot_contract` 下登记 `pg363_pooled_rule_ir_candidate`；JSON 解析通过，最新 rules SHA=`a5f9dbb93db3527d21aff485d6ed106fe7e941fa6dfa8a60e50c66e7351d65a2`。下一动作是第二独立动态实现上的 hard-negative/ASK/failure→repair 与“模型选槽→evaluator 生成真实 wire→fresh typed replay”，不是把原始攻击字符串塞入模型。

### 2026-08-09：PG-364 组合实现留出、微批次修复与熵塌缩对照

- PG-362 的原始 implementation holdout 同时引入了大量训练中未出现的目标值，无法单独解释组合泛化。新增 `scripts/build_pg364_compositional_rule_ir_dataset.py` 与 `scripts/audit_pg364_compositional_rule_ir_dataset.py`：只用 source sidecar 的 implementation 做 split，随后只输出原有抽象 context/13-slot target；实现组只保存 salted hash，不进入模型。PG-364 训练=`1600` 行、implementation holdout=`1000` 行、train implementation groups=`13`、holdout groups=`5`，组完全不相交；每个 holdout slot value 在 train 均出现，7 轴和 621 token 窗口完整保留。
- dataset=`research/pg364_compositional_rule_ir_dataset_v1.json` SHA=`9eb16a89e7ca813b42f1a473945ea44ff7ac9a39964c23e07e169aa0d58fd618`；audit=`research/pg364_compositional_rule_ir_audit_v1.json` SHA=`c1b7fa04a6d298202d5bd6cec11d71171147d82a4d4948654c761d2b7930c7f2`；builder SHA=`a670451d02ec79f5e7b69026f71d907859684aaad395f6dc25e7643b616bcc8d`；audit script SHA=`634f5c765acf46f10a88fb0b8fe5a05441079d9b3ffaedaacde074a5194f7f2a`。
- 第一次 PG-364 A800 启动暴露了工程问题：全量 attention batch 会在第一步 OOM（A800 80GB 仅剩约 7.6GiB）。`app/pg363_pooled_rule_ir.py` 现改为可控微批次训练/分批 slot 推理/分批 entropy，默认 batch=`16`；这不是降低窗口，而是避免把所有记录同时展开成 `batch×sequence²`。相关 focused 回归=`19 passed`。
- 周末远程 A800 GPU0（`CUDA_VISIBLE_DEVICES=0`，GPU1–7 未触碰）完成 3 seed×16 epochs candidate-only。report=`research/pg364_a800_compositional_rule_ir_candidate_v1.json` 文件 SHA=`763110f248e124ce125604b3832f4dc5d3be3a48347ae64fffe760986cb3d73e`，内部 SHA=`6719009a1b28f1addf560a08b3001b2f81b8e6df8a3bd936836fe590e09e1f1e`；checkpoint=`artifacts/pg364-compositional-rule-ir/pg364_a800_compositional_rule_ir_candidate_v1.pt` SHA=`9e06ad9ed85ce0d25d0ed7a0dd11f394542b0cbe9940e7f4663b23ff84177d43`。训练后 GPU0=`1 MiB/0%`、无 compute app。
- PG-364 三个 seed 的 holdout 槽位/完整 Rule-IR/ASK/repair/abstain/positive/negative 指标均为 `1.0/0 false allow`，但 predictive entropy drop 分别为 `0.839885/0.833947/0.846737`，最坏=`0.846737`，熵门失败。解释是模型学成近似确定性映射，不能用完美分类覆盖信息塌缩；status=`blocked_candidate_only`，训练、长期记忆、payload catalog 和漏洞声明全部关闭。
- 当前下一实验是 PG-365：在同一 PG-364 split 上加入只作用于训练 logits 的 label smoothing/entropy-preserving 对照，并重新看最坏 seed；不能改 holdout、删 token 或用安全 guard 掩盖熵下降。最新 rules SHA 将在 PG-365 运行前重新锁定。

### 2026-08-09：PG-365 熵保真对照与经验固化

- PG-365 是 PG-364 的单变量对照：同一 dataset/audit、同一 1600 train + 1000 implementation holdout、同一 621-token window、同一 3 seeds=`36301/36302/36303`、同一 A800 GPU0；只增加 `label_smoothing=0.1`，没有改词表、split、模型输入或 evaluator 边界。decoder SHA=`7e05f7449a76124038542eb22b63ccd5d36ff341f276a658bcca92155adff80f`，runner SHA=`5671aefae6f449c0549de31244c89179013579e19b7944f8ee28c189cd8e5fe1`，训练前 rules lock=`86c06a7408f7ef54ff121cf378aac65ed270e4805ba2070e44f8eea8e82b25a6`。
- 远端 `112.111.7.91:60228` 显式 `CUDA_VISIBLE_DEVICES=0`、A800-SXM4-80GB GPU0 完成 16 epochs、batch=16；训练后 GPU0 回到 `1 MiB/0%`，没有 GPU1–7 操作证据。report=`research/pg365_a800_entropy_preserving_rule_ir_candidate_v1.json` 文件 SHA=`fbfe187ddfd2ea4caabcee7a12affa2d8d0d8b8ab9c14789bc6358ffcd5ec48`，内部 report SHA=`4f2f05eece9733e9a87f142fac1534ee3475b9294c8c73c20de9ca027c222d32`；checkpoint=`artifacts/pg365-entropy-preserving-rule-ir/pg365_a800_entropy_preserving_rule_ir_candidate_v1.pt` SHA=`ae362a940f8f91ebbf02f21ac7b9008bc7579756a27abc5a6435e94b36b500c1`。
- 结果：三 seed 的 holdout predictive entropy 相对下降=`0.761890/0.755098/0.767149`，最坏=`0.767149`，仍远超 `0.25` 熵保真门；但 implementation holdout 的 ASK/repair/abstain/positive recall 和 Rule-IR exact 均为 `1.0`，negative false allow=`0`。因此这是 `blocked_candidate_only`：能力分数好看不能覆盖严重信息塌缩，label smoothing 只改善了 PG-364 的 `0.846737`，没有解决根因；promotion/training/memory/payload/vulnerability 全部 false。
- 经验规则：熵不是越高越好，也不是越低越好。高熵且能力差通常表示没学会；低熵且能力好但相对下降超过 25% 通常表示塌缩/记忆化；只有能力、安全、负对照、校准和熵保真同时过门才算进步。以后每个候选必须同时保存 baseline/post entropy、relative drop、worst-seed 能力指标和 failure interpretation；禁止降低熵阈值、只报漂亮准确率、或用 guard/平均值替代信息保真。下一步应优先检查表示可辨识性、目标分布、hard-negative 与跨实现数据，而不是盲目增加 epoch、容量或词表。
- 本轮本地 focused regression=`7 passed`；远端训练产物已复制回本地并核对 SHA。rules 文件更新后最终 SHA=`fc8778d064d6c13662c6e038490d6e036dea4d7d0b5b4fe473349ff936957a95`。当前没有运行中的远端训练、Docker 容器或本地训练进程。

### 2026-08-09：PG-366 上下文可辨识性空目标对照

- PG-364 的组合数据在完整抽象 context 下存在 shortcut 风险：2600 行、train=1600、implementation holdout=1000，精确 context group=`840`，exact conditional target entropy=`0`，unique context ratio=`0.323077`；presence-only 投影的条件熵约 `6.671` bits。`source_meta` 在 split 清洗后为 `0` 行，不能把实现身份误当作模型输入。
- 审计器 `scripts/audit_pg366_context_identifiability.py` SHA=`3f0a6a3f09a77d2f523a27db049068d09685ecaa5e8ce3cb0e080f2f4f85e50a`，报告=`research/pg366_context_identifiability_audit_v1.json` SHA=`fece2332ab5021c47aaf9292c880fa88064881c4aee77e0dc4a6c57c180402a0`，status=`diagnostic_shortcut_risk`；该审计不授予训练/记忆/漏洞晋级。
- A800 GPU0 空目标对照：runner=`scripts/run_pg366_a800_context_identifiability_null.py` SHA=`ba01b75e591eb64203fed6b93f53a270eaab317c790567cb526f2e6cf7f18841`，保持 context 不变、只置乱 train target，远端 A800 GPU0（`CUDA_VISIBLE_DEVICES=0`）完成 3 seeds。报告=`research/pg366_a800_context_identifiability_null_v1.json` 文件 SHA=`7d8ae47c32dc8e94cb6c08ba097ceff8b026555f08115fe30ea633b01f59e171`，内部 SHA=`6f29d9600d60c0fadf8dda18a693eb43ae890633c1b0a3750b9d04bde5aca0a2`，checkpoint SHA=`b8be57f2a71a173e09f8c27ebc09e5fb8c4118d262be3d659a94cba879f885a6`。
- 空目标结果：holdout sequence exact=`0.0`，negative false allow=`400`，最大 entropy drop=`0.352547`，status=`diagnostic_null_control_only`。它确认正常高分依赖 context-target 相关性，但也说明 PG-364 目标映射过于确定；不能据此宣称模型学会了主动探测或 payload。

### 2026-08-09：PG-367 WAF 阶梯过程轨道（诊断候选）

- 用户要求的过程已单独建模：`probe→surface/vulnerability hypothesis→reproducibility uncertainty→WAF filter observation→one-variable repair→typed effect/negative→fresh replay`。`app/pg367_waf_staircase.py` SHA=`5863a6303b1e5c9084f938442c5ed417c3ced8fe4e0a6766afb72a793d2ec385`，支持六种抽象策略：allow、delimiter normalizer、pattern rejector、decode-once、length cap、parser boundary。
- 前后端 loopback runtime=`app/pg367_waf_runtime.py` SHA=`43e457884ba87e24a34835b8664c024512bdc5938f50f34fcd58ae376b2b3c2b`：只绑定 `127.0.0.1` 临时端口，HTML 页面和 GET/POST API 均返回有界 projection；请求具体值、原始响应、外网、持久化写入均不保存。
- 抽象数据集 builder=`scripts/build_pg367_waf_staircase_dataset.py` SHA=`dc3cb36d5a4624fb67a6521a335a9dec26c583e1d4af5e6c4ce03bc2eaeebb75`；dataset=`research/pg367_waf_staircase_dataset_v1.json` SHA=`42ad8504a2beac696f430b7089fca489f2fba1a47a2786dd044ef9a5b4c39211`，352 rows（train=224、implementation holdout=128、GET/POST=176/176、failure=160、repair=160、candidate/reference/negative/replay 四角色）。
- 新增审计器=`scripts/audit_pg367_waf_staircase.py`，报告=`research/pg367_waf_staircase_audit_v1.json` 文件 SHA=`E85D23F89042DE1757B8A778E9E674A121D72A88ED10E00EFF7B5DCBF7D9B5C0`，内部 audit SHA=`a67db99deccf9cc9ec797017c9f846fb004893a7dccca2afeb2bc1f89e67c6ab`，status=`passed_diagnostic_only`。审计确认 raw hits=0、repair action-change=160/160、negative clean=88/88、GET/POST 覆盖完整；`training_eligible_rows=0`，promotion 全关。
- 重要边界：这批数据证明“WAF 过滤反馈和失败修复过程”可被采集，不证明通用网址存在漏洞，也不证明神经模型已经生成可迁移原始 payload。原始字符串只能由授权 evaluator 最后一跳临时绑定，模型上下文仍只有抽象 WAF/Rule-IR token。
- 真实 loopback replay 已完成：runner=`scripts/run_pg367_waf_staircase_replay.py` SHA=`dfa05f6e0a71a6938aabc732062f52d565adbb1aa8a5478d7f68fc0a15c700bb`，tests SHA=`7d4ff08004d209263ecd7ce6b2312d44c25a0b95437a9e4db2f06452cddf4faa`，报告=`research/pg367_waf_staircase_replay_report_v1.json` SHA=`2d3f00240aff59bc61c3a422dedde6a49fcf067c9b29e9c782646eeae797994c`。六策略×GET/POST=`12` episodes，candidate/reference typed=`12/12`，negative clean=`12/12`，negative violation=`0`，失败→单轴修复=`40/40`，fresh reset/evidence/replay=`12/12`；实际 loopback GET/POST body 只在进程内使用，持久报告仅为哈希/有界 projection。`--show-wire` 仅临时打印一次性 loopback canary，绝不写入报告。该结果只证明 evaluator 闭环，不证明神经模型选出的 wire。
- replay 已登记到 `research/improvement_rules.json`；在加入 `--show-wire` 临时人工复核开关后，当前规则外部 SHA=`efbf9f61f3d42c633af907d7571f9c8a36a26630cc3d1cb6359c493283733c2f`。在第二独立实现、模型选槽→binder、负对照和 predictive entropy 共同通过前，不启动 PG-367 capability 晋级训练。

### 2026-08-09：PG-367 组合留出高容量 A800 对照

- PG-367 v1 的整策略留出把未知 WAF 值和组合泛化混在了一起（首轮 A800 448 个 holdout unknown token、sequence exact=0）；该结果保留为失败对照，不修改、不覆盖。
- v2 builder=`scripts/build_pg367_waf_staircase_dataset_v2.py` SHA=`d7ea5e6a82294bcdf05349214ca2529f1d9969a653e7d5d583a7b50a74f503f1`，dataset=`research/pg367_waf_staircase_dataset_v2.json` SHA=`aef0788f65b8870bd5ee2a26419e876589d4d8ac4af39cc0ef5f5a97d1df4913`；保持 352 条、GET/POST=`176/176`、failure/repair=`160/160`、四角色和 WAF 轴不变，改为确定性组合留出 train=`275`、holdout=`77`，holdout 未出现训练外 token（unknown=`0`）。v2 audit=`research/pg367_waf_staircase_audit_v2.json` SHA=`060b1f7ba7e0573f611df2c89f780166fc3e422f9efa4245cf97ddd2637c35e5`，status=`passed_diagnostic_only`。
- 周末远程 A800 GPU0 高容量 next-token candidate：runner=`scripts/run_pg367_a800_process_candidate.py` SHA=`431958637d3a465c69fc519c07b7f68fab59277c025c2b2adf8d51886c496920`，3 seeds=`36701,36702,36703`、16 epochs、d_model=`256`、4 layers、4 experts、hidden=`512`、batch=`32`；明确 `CUDA_VISIBLE_DEVICES=0`，GPU1–7 未触碰。report=`research/pg367_a800_process_candidate_v2.json` SHA=`70b571fc650a9fa876d95efcc327c74479c080d555ea63751b6404a2203d1a01`，checkpoint SHA=`8cbad643578261da2a4cc4b82026231e9eb31dd4305cc6a7b4c7fbc38356002a`；checkpoint 已复制到 `E:/blackboxanalyze-archive/artifacts/pg367-a800-process/`，哈希一致。
- v2 结果：holdout sequence exact=`0.662338/0.688312/0.662338`，最坏=`0.662338`；unknown token=`0`。这说明修正 split 后高容量 next-token 能学到部分 WAF 过程组合，但没有 typed oracle、负对照、fresh replay 或模型选槽→wire 闭环，single synthetic implementation，status=`blocked_candidate_only`，promotion/长期记忆/payload/vulnerability 全关。
- 在同一 v2 split 上做了关键 Rule-IR 槽加权 SFT 对照：runner=`scripts/run_pg367_a800_rule_ir_sft_candidate.py` SHA=`ef5b943618bb6aeac92a5d91e3d425c97c6ea683d289e159d3e2cf0449dc1d38`，critical slots（ASK/next_action/repair/oracle/safe）权重=`3.0`、其它上下文=`0.25`，同为 3 seeds×16 epochs、d_model=`256`/4 layers/4 experts。report=`research/pg367_a800_rule_ir_sft_candidate_v1.json` SHA=`7c692811a382ea2edaa3d754ae823c8ab9e21d6b0c0aebf0b5e51bf883a30dc8`，checkpoint SHA=`a78885c781f02e0cae22a6b5bd267982eb33f3cec3cd160fb21ce764086443a2`，已复制至 E 盘归档。
- SFT 结果：holdout sequence exact=`0.532468/0.597403/0.441558`，最坏=`0.441558`，反而低于 plain v2 的 `0.662338`。这说明“关键槽加权”不能直接替代完整序列与跨组合校准；它是有价值的失败对照，不晋级、不进长期记忆。当前规则外部 SHA=`525a7149b159c09fe524fc6ccf0f637a64ffca6053543cf3b76d73ee53657c28`。
- 规则文件在登记 v2、replay 与 `--show-wire` 后外部 SHA=`efbf9f61f3d42c633af907d7571f9c8a36a26630cc3d1cb6359c493283733c2f`。下一高上限动作是把 v2 模型的抽象 Rule-IR 输出接入同一 allowlisted binder，再在第二独立动态实现做同样的 fresh GET/POST/negative 复放；不以 A800 分数替代 typed fresh 复放。

### 2026-08-09：PG-367 模型选槽→binder→fresh 四角色回放

- 新增 `scripts/run_pg367_model_binder_replay.py`（SHA=`d3fdb9265eda0957f437837a1439b27b6e711b3b062e20d453c8950dc5e5fff8`）和 tests SHA=`56fe3d1b90fe2a46e7c8b175a1755e1e67fd74a94a04348dbf48bd76409c8819`。它从 PG-367 v2 A800 checkpoint 解码 holdout target，只保留抽象 Rule-IR 槽；通过 `app/pg350_runtime_payload_binder.py` 的 source-attested template 后，在 PG-348 loopback 动态 runtime 做 candidate/reference/negative/replay fresh reset。模型若 `safe_to_send=0` 或槽不合法则 abstain；negative 行的 unsafe allow 在 binder 前阻断并计数。
- 报告=`research/pg367_model_binder_replay_report_v1.json` SHA=`633f6f81300233f2f1c6cd2a3bbbab7f1e7a39a2e14580863205566ae51cad93`，checkpoint SHA=`8cbad643578261da2a4cc4b82026231e9eb31dd4305cc6a7b4c7fbc38356002a`，registry SHA=`a500b3edfaed697f07ec6551cc5ef0e6125682b7a9ad386130d4caf8f14ce855`。77 条组合留出：decoded exact=`51`，模型可绑定=`35`，confirmed positive=`35`，安全 abstain=`42`，unsafe allow=`0`，safe abstain=`42/42`。这是真正的模型选槽→binder→typed replay 证据，但仍只有单一合成 PG-348 实现，不能宣称通用漏洞或原始 payload 能力。
- `--show-wire` 只在人工复核时临时打印 local canary wire；报告仅保存 slot、template/source/request/response/evidence 哈希和有界 projection，raw URL/body/response/canary 不进入模型或持久记录。
- 当前规则外部 SHA=`483c8b9348941f431ab093d850613c9905cef8937e5f5b8b4c78b4192fe09c00`。下一高上限动作：第二独立动态实现的同一模型选槽回放；若跨实现 hard gate 失败，保留失败轨迹用于下一轮 SFT/RL，不继续在单一 synthetic lane 上堆容量。
- 本轮资源复核：A800 训练后 GPU0=`1 MiB/0%`、无 compute app，GPU1–7 未触碰；C/D/E/F 剩余约 `65.8/156.6/152.4/89.8 GiB`（C 盘工具显示约 70.6 GB）。PG-367 v1/v2 checkpoint 已复制到 `E:/blackboxanalyze-archive/artifacts/pg367-a800-process/` 并核对 SHA；未删除任何文件，保留 D 盘工作副本与可恢复归档。

### 2026-08-09：PG-368 第二实现计划与展示投影

- 第二实现计划已由代理完成：`scripts/plan_pg368_second_implementation.py` SHA=`2cc24483bbbe1c9f4963cb8b036a5ac21ca710b8b78a245719c3405b648f55d1`，测试=`tests/test_pg368_second_implementation.py` SHA=`22b352bb74097dfd6abf04085687442150fe426fac58834c8beafeea31caa0bb`，计划=`research/pg368_second_implementation_plan_v1.json` SHA=`a4bc8bc2ecefa71d634d73a2e8a770a8afd810c5c17a846b0bfe38cfb340d811`（内部 `plan_sha256=27f4bfba5c3e83cf282a4935600c4f8ed5582e9019f1682eb5f42078121b8263`），只读 audit=`research/pg368_second_implementation_audit_v1.json` SHA=`84372fac49aa1bf7436b161bd468585879dd14b2cff39f2567da6bd271a4d365`（内部 `audit_sha256=573b8372b56fafbc2960d5986b0dfc7d48f4a55bd8b60b5646f230d35fbd6f1`）。固定 WebGoat image=`webgoat/webgoat@sha256:3101bd9e7bcfe122d7ef91e690ef3720de36cc4e86b3d06763a1ddf2e2751a4b`，使用不同的 Docker relay/进程边界；3 seeds×GET/POST×candidate/reference/negative/replay，network-none、loopback、无端口/挂载、每角色 fresh/reset/evidence SHA。当前 `planning_only`，未启动 Docker、未接触目标；缺 typed evaluator 时模型投影统一 `ASK/safe_to_send=false`。
- PG-368 展示投影已完成：`app/pg368_capability_projection.py` SHA=`ed6a21bb68bb41ac96228b7557403f2cb63175e1d20c7ac01a7b72b76bfea126`，测试 SHA=`e05a4149d2433e5126d4fbf2d30bea4d30d804dfdd9a876a794d253cd222a50c`，artifact=`research/pg368_capability_projection_v1.json` SHA=`3b33ec315ac51b10f69cc2bc97884150b683c0beab8c9d7e5ac35d851424362a`。展示摘要：模型选槽 holdout=77、decoded exact=51、bindable=35、confirmed positive=35、safe abstain=42/42、unsafe allow=0、evidence/fresh/negative=35/35/35；WAF evaluator episodes=12，GET/POST 各6，candidate/reference=12/12，negative violation=0。该证据仍是单一合成实现的 evaluator-only 工程闭环，不是通用网址漏洞能力。
- 当前最短下一步：在 WebGoat 计划通过显式 `PG368_LOCAL_DOCKER_EVAL=1` 后补一次真实 fresh GET/POST method-shape replay，再把同一抽象 Rule-IR binder 接入；只有模型选槽在第二实现上通过 candidate/reference/negative/replay 和 worst-seed 门，才讨论新的 A800 capability 训练。未触发的细节按 fast-lane 记为 diagnostic/ASK，不得降硬门。
- PG-368 binder 已完成但默认保持 dry-run：`scripts/run_pg368_webgoat_binder_replay.py` SHA=`fb06d2f5cc6f50772015f12a6d92d62d901ad671e82553d6d63647a7fe906dbc`，测试 SHA=`86bd55c24b0875fdf0cea09f03ce533725b26844baae88a181524e5709236a2d`，report=`research/pg368_webgoat_binder_replay_report_v1.json` SHA=`fd58fde2a0a384401a4494feebac5c1bff4599b25345eba44c1674f6256e51ea`，内部 `report_sha=3f49b826e7ec7bf68d30a25d8820ba11b08fcef637dc163f1f8f7d1fc0e67838`。6 episodes×4 roles=24，24/24 ASK、target_contacted=0、typed/confirmed_positive=0、unsafe_allow=0；只有 `--live`+`PG368_LOCAL_DOCKER_EVAL=1` 才能进入结构性 WebGoat evaluator，仍不产生漏洞或 payload 晋级。
- 本次登记后的 `research/improvement_rules.json` SHA=`1712825b5bcfe29eae7cb8678d34e383f47c688b83fe0811e67600f94c9889e9`；AGENTS 自身哈希只在外部校验时计算，不写回备忘录。

### 2026-08-09：PG-368 单 seed live method-shape smoke（能力仍阻断）

- 在固定 WebGoat digest、`PG368_LOCAL_DOCKER_EVAL=1`、`--network none`、loopback relay、无端口/挂载的条件下，完成了一个只读单 seed smoke：`research/pg368_webgoat_binder_replay_smoke_seed36801.json`，文件 SHA=`585fef59ae56bf3e6b37ee18aa6c9c8a44b1cb7ef304599c0e14920e1d30b140`。
- 结果：2 个 route（GET/POST）×4 角色=`8` 个 fresh disposable 容器；`target_contacted=8`、结构性 typed method-shape=`6/8`、`negative_violation=0`、`fresh_role_reset=8`、`model_selected=0`、`ask_rows=8`、`confirmed_positive=0`；回放后匹配容器=`0`，持久报告不含 raw request/value/response/wire。
- 这是 evaluator 工程验证，不是模型能力或漏洞结果：PG-368 模型投影仍统一 `ASK/safe_to_send=false`，POST/GET 结构观察不等于漏洞存在；未生成训练行、payload catalog 或长期记忆。
- `research/improvement_rules.json` 已把该 smoke 登记到 `pg361_payload_shape_syntax_slot_contract.pg368_second_implementation_plan.pg368...live_smoke`，当前规则外部 SHA=`c38a0a99d76db7a0587ea738e46bb1693315053fd294ce4ac8b9a0ef58607242`。后续必须做完整三 seed、模型选槽→binder、正/参考/负/回放和跨实现 hard gate，才可申请 A800 capability candidate；不以这次 6/8 结构分数冒充 payload 能力。

### 2026-08-09：PG-368 槽位覆盖审计（训练门继续阻断）

- 只读代理审计了 PG-368 计划 24 role contract、PG-333 WebGoat 18 rows、PG-337 DVWA 9 rows、PG-342 WebGoat 6 rows；没有修改旧数据、没有启动 Docker/GPU/网络。
- 四组均缺 `syntax_category_ref` 与 `payload_shape_ref`（观测为 `0`）；已有 `js_syntax_shape=empty` 是单一占位值，不能当语法类别。`encoding_chain` 和 `parameter_role` 只保留为安全抽象投影，覆盖 57 行但形状过窄；typed oracle/evidence 只能留 sidecar，不能进 context。
- failure→repair 只在 PG-337（9 行）和 PG-342（6 行）存在；PG-333 是 clean method-shape，PG-368 计划仍全 ASK。PG-367 v2 已有这两个 slot 名作为 schema 对照，但不能回填旧行或把 `query_marker/html_form_marker` 冒充未见实现观测。
- 规则登记：`pg361_payload_shape_syntax_slot_contract.pg368_second_implementation_plan.slot_coverage_audit.status=blocked_missing_payload_shape_slots`，`training_eligible=0`；只有补齐 source-grounded 两个 slot、GET/POST 多形状和独立 implementation holdout，才进入 A800 capability candidate。当前规则外部 SHA=`a3314bf5ae51e02b93d7cee14f97f01d53fb1b12c6b237556342ce47df5d0c02`。

### 2026-08-09：PG-368 槽位覆盖审计脚本冻结

- 审计脚本=`scripts/audit_pg368_slot_coverage.py` SHA=`51b869f22d9da1ef4a62eb182b5c41a94edafc834207662a27dcf6263a4e9599`；测试=`tests/test_pg368_slot_coverage.py` SHA=`5c22796a45c109876d23d4d8b2d6047853cddb16e731f99707caa6e31691659f`，focused=`6 passed`。
- 只读报告=`research/pg368_slot_coverage_audit_v1.json` 文件 SHA=`88185f57294559ba96f2130922d5d666ffbd533ab386015990e7a0a4da1c7b26`，内部 `report_sha256=257a1b308068f6e3b3f40d1fc52d7d11f7d69e4b8edb0f9ff9efb71d62210885`，status=`blocked`；报告只保存计数/哈希/抽象覆盖，不复制原始 probe、请求、响应或 route literal。
- 报告把 PG-367 v2 的 7 个 target slot 当作 schema 参考，审计 PG-368 plan/PG-333/PG-337/PG-342；四组 `syntax_category_ref=0`、`payload_shape_ref=0`，没有旧行重分类或训练样本生成，promotion 全关。审计脚本的 blocked exit 是预期行为，不是运行失败。
- 规则登记后外部 SHA=`9abfeb72c469de1a9797cbce018675202de1af15b4d7ba29ad9561ce4cf98aae`；下一步只允许补采真实 source-grounded slot，再申请完整三 seed/跨实现 A800 candidate。

### 2026-08-09：PG-369 A800 动作均衡高容量候选（失败对照）

- 训练前把 PG-351 v2 抽象数据、审计、代码和当前规则复制到独立远端目录；远端使用 Anaconda CUDA 环境，显式 `BLACKBOX_REMOTE_A800_TRAIN=1`、`CUDA_VISIBLE_DEVICES=0`，未触碰 GPU1–7。
- runner=`scripts/run_pg351_a800_ask_oracle_composition_candidate.py` SHA=`9f83bbc9ab870983d54994ad54b0bbb0c97a0d84dd5919b43faa407f17e043a3`；数据 SHA=`f127122fd53f39a382ff7d7f73c38267217c214114d7344735c900b832b15202`；audit SHA=`a1ddf7950870bbf6260cb8cecf5ff20c7f17c083d4de0b3c756a5eac44ef7bd5`。配置为 d_model=256、4 层、4 experts、hidden=512、batch=16、8 epochs；动作均衡，repair/abstain/replay/safe 权重分别 12/10/10/8/6，受限 Rule‑IR 解码开启。
- report=`research/pg369_a800_actionbalance_candidate_v1.json` 文件 SHA=`3aa86fa92b30e90a1c659082c1f56f77a282fa5c59e9399ef335cd4cbd429ae7`，内部 SHA=`41068752e960cca4526a2eec8f3457013a2ee1271bac85d0448b5f3626e2b084`；checkpoint SHA=`b8d02dac7521ddca771a09e06357d05b176bd337ddafc97e02452c7eb0d06f75`。
- 结果：ASK 最坏=1.0，但熵相对下降最坏=`0.881314`、负例误放=`507`、repair=0、abstain=0、positive-action=0、positive recall=0；状态=`blocked_candidate_only`。结论是动作复制/强权重制造了确定性错误路由，不能继续加 epoch、扩大容量或晋级。
- 下一轮改为不复制动作行的共享 backbone + 独立 next-token/slot/ASK/repair/negative heads；仍保持 raw/context firewall 和所有 promotion=false。

### 2026-08-09：存储与容器回收

- 只读核对余量：C=`62.96 GiB`、D=`156.02 GiB`、E=`152.32 GiB`、F=`89.80 GiB`。未发现可按 SHA-256 安全删除的重复权重；E 盘历史归档和 PG-367 D/E checkpoint 副本保持不动。
- 在逐容器确认 `Exited` 且无挂载后，回收 6 个旧靶场容器：`sift-loop12-juice-v20`、`sift-pg146-juice`、`sift-pg146-dvwa`、`sift-pg146-webgoat`、`sift-pg25d-vulnerableapp-cycle1`、`sift-pikachu`。保留有 bind mount 的 `sift-loop12-proxy` 与无关的 `chatgpt2api`，未执行 broad prune；当前 `docker ps` 无运行靶场。
- Docker 可回收空间约 containers=1.7MB、volumes=625MB、build cache=1.706GB；未自动删除未标记 volume/cache。当前规则外部 SHA=`8915040d214b9746f446ccb50f568454c7e9fb7a578916afbadc3ff75e30d67f`。

### 2026-08-09：PG-370 共享骨干多任务 A800 候选（信息门失败）

- runner=`scripts/run_pg370_multitask_moe_candidate.py` SHA=`41fdcd50b0a4082cfcb2fced838883683be6f6b8b79dfbd2e8ca1d60f652b69a`；tests=`tests/test_pg370_multitask_moe_candidate.py` SHA=`29768ea7b245f263b6e030380341740144d6e7e488e33dbbcc8251e1ec71fc67`。共享 PG-295 CausalMoE backbone 加独立 13-slot、ASK、repair、negative heads；target-mask 从 `context_len-1` 起覆盖首个 target 与 `TARGET_EOS`，避免 next-token 目标被截短。
- 严格 train-only 词表审计发现 implementation holdout 有 23 个训练未见抽象 token，`encoding_ref` 与 `syntax_category_ref` 各有 1 个新值；默认 train-only 模式阻断，不能把留出行偷偷并入 vocab。A800 candidate 明确使用两个数据集预先声明的 append-only ontology inventory（vocab=877，required window=621，max_length=768），报告记录 `vocabulary_scope=declared_ontology_manifest`，不把该路径冒充纯 train-only 泛化。
- 远端：`112.111.7.91:60228`，A800-SXM4-80GB GPU0，`CUDA_VISIBLE_DEVICES=0`；3 seeds=`37001,37002,37003`，4 epochs，microbatch=16，d_model=256，4 layers，4 experts，hidden=512。GPU1–7 未触碰；训练后 GPU0=`1 MiB/0%`、无 compute app。
- 报告=`research/pg370_multitask_moe_candidate_v1.json` SHA=`06699e3551b00606b3d13a5ecb7760d46803485a905341335399defc178e519d`；数据/审计锁：PG362=`2f2e270a9143e9488a7e9206cace944abb3b0f37b31bd8b3059bf2bc3c3f4d35`/`2d1d76179575d0b5de80442a05b094def02a3d02e76f5f38cc719b0a7acfb081`，PG367=`aef0788f65b8870bd5ee2a26419e876589d4d8ac4af39cc0ef5f5a97d1df4913`/`060b1f7ba7e0573f611df2c89f780166fc3e422f9efa4245cf97ddd2637c35e5`；报告 rules SHA=`8915040d214b9746f446ccb50f568454c7e9fb7a578916afbadc3ff75e30d67f`。
- checkpoint SHA：seed37001=`2b86eefb2e21ec977ba1647688bfc9e7ba4011d18e24828b241d10fd8ba49f00`，seed37002=`74f535bc3f9e2f9853781128a8b91675ca2a45fd0d9dd1073771c2c6f5dc0009`，seed37003=`e8bf0f10b4c598cb62411d1fe1616559e1c30d9b73b081f0442e413e5136dc38`。checkpoint 仅含抽象模型状态/词坐标，不含 raw wire、响应体、evaluator 答案或 payload。
- 结果最坏 seed：sequence exact=`0.020311`，slot accuracy=`0.726368`，ASK recall=`1.0`，repair recall=`1.0`，positive recall=`1.0`，negative false allow=`0`，predictive entropy relative drop=`0.803455`。熵门（≤25%）失败，状态=`blocked_information_entropy_candidate_only`；多任务头学会了安全边界，但完整 Rule-IR 组合没有学会，且分布塌缩约80%。
- 结论：PG-370 是失败诊断候选，不进入训练晋级、长期记忆、payload catalog 或漏洞声明；不要继续用加 epoch/加容量覆盖表示问题。下一动作是补齐第二动态实现的 source-grounded syntax/payload-shape slots，做跨实现模型选槽→allowlisted binder→candidate/reference/negative/fresh typed replay，并优先修复组合序列与熵塌缩。
- 本轮 focused 回归：PG-370/PG-369 及相关数据测试=`18 passed`；当前无远端训练、Docker 或本地训练进程。规则文件本轮更新后的外部 SHA=`f6011f8f643da72244d729ff5fdb2823f17c3749ee304c63197f946e3945dbdc`。

### PG-370 研究台接入补充

- 研究台只读 projection 已接入 `app/research_ops.py`，当前 SHA=`9393b74ff8469095493f00cdbc7e5b99e89bcaf9a534abf4c79276754c89139c`；专项测试=`tests/test_pg370_research_ops_projection.py` SHA=`62b5ea3cf33a711a6433c781bdc67194a691cb205a95d7aabb4a9589d8dfa4a2`。
- 研究台仅展示 vocabulary scope/gap、required window、worst-seed sequence/slot/ASK/repair/negative/entropy、A800/checkpoint 哈希前缀；缺报告显示 pending，原始 rows/tokens/checkpoint 内容/wire 不出现在 snapshot，promotion/memory/payload/vulnerability 全 false。`tests/test_research_ops.py` 与专项测试共=`63 passed`。

### 2026-08-09：PG-370 研究台重复投影清理与最终证据

- 已移除重复的 `pg370_multitask_moe` 投影块，只保留唯一的 `pg370_multitask_moe_candidate` canonical projection；当前 `app/research_ops.py` SHA-256=`fe7c7aea0b99bd8d96f3a82025bffaeed6fdfe6c9ac1078fdfb4329a7202e399`。
- 专项测试 `tests/test_pg370_research_ops_projection.py` SHA-256=`62b5ea3cf33a711a6433c781bdc67194a691cb205a95d7aabb4a9589d8dfa4a2`，研究台、投影和 PG-370 回归共=`72 passed`。
- PG-370 报告文件 SHA-256=`06699e3551b00606b3d13a5ecb7760d46803485a905341335399defc178e519d`，内部 `report_sha256=bc289df3c694916a3d4c24505132f5b6bc3bcc4c7deb70775aa1a6864575d25d`；最坏 seed sequence exact=`0.020311`、slot=`0.726368`、ASK/repair/positive=`1.0/1.0/1.0`、negative false-allow=`0`、predictive entropy drop=`0.803455`，超过 25% 信息门，仍为 candidate-only，promotion 全关闭。

### 2026-08-09：PG-371/PG-372 绑定合同与信息修复审计

- PG-371 binder contract 已冻结：`app/pg371_model_slot_binder_contract.py` SHA=`caf08148c9991a807dead76a793f06a6458fd71e2195b8eb984515580b8c18bb`，planner=`scripts/plan_pg371_model_slot_binder_contract.py` SHA=`81892fe332fea5171f178bbc231ab1bf09bc1f28b35d47d3548163a9555fdf16`，tests SHA=`8d53d69ded49ea241b457ff2923867fdcacc353f11828fa1e746b7c136d86add`，plan=`research/pg371_model_slot_binder_plan_v1.json` SHA=`78ab81d1a3aa3c8f166f2f128dfbd89c9cd5db3dd4f915c1cdb1635f4faa5ed0`。3 seeds×GET/POST×candidate/reference/negative/replay=24 个 planning rows；模型选槽、typed effect、wire、target contact 全部为 0，缺证据统一 `ASK/safe_to_send=false`。13-slot Rule-IR 经过 8-slot binder gate，不能自动生成任意字符串或升级成漏洞结论。
- PG-371 信息审计=`scripts/audit_pg371_representation_entropy.py` SHA=`928b2e860db9249192d3bb3c114247c200a7b2573c1f0e6999b71ec3ee9e842c`，report=`research/pg371_representation_entropy_audit_v1.json` SHA=`93b06e7b8e55232c92496f93e6ce3558c8832d7a26e502eb9e996e89ef7c8793`；PG-370 的随机/未训练熵基线被判无效。PG-372 builder=`scripts/build_pg372_failure_repair_dataset.py` SHA=`c862abbca840df406760e68bb475dacb832de81e5118b2c9980c622710b31224`，dataset=`research/pg372_failure_repair_dataset_v1.json` SHA=`344fe9762bd098786cf7dfc4ae14547f7a7a29657b23e607a90210d13471f9e7`。holdout-precedence 后 repair dataset=1072 行、pair=102、holdout unknown context/target=`21/2`，训练资格仍为 0；不回填、不晋级。

### 2026-08-09：PG-373 分阶段 A800 候选

- PG-373 runner=`scripts/run_pg373_staged_pretrain_candidate.py` SHA=`6c8ca751b8f90cf505045e29ac841e694f431dfcc9598b971658cc0209efe7c4`，tests SHA=`66f43da59c121e274d56c9872eb6121bc54df8cf5117952519662e8192ca80fd`，report=`research/pg373_staged_pretrain_candidate_v1.json` 文件 SHA=`0e689f20482e3ceb14ffe66a240587fa7336d069835ac23b4cb4739c5c25b6ef`，内部 report SHA=`80065cbaecfc767c769b8ad11fa083f28a4f868cd746dbfec25aad226e96465e`。先 train-only next-token 预训练 4 epochs，再用低学习率/KL 锚定训练结构化 Rule-IR/ASK/repair/negative heads；A800 为 `112.111.7.91:60228` GPU0，`CUDA_VISIBLE_DEVICES=0`，GPU1–7 未触碰。
- 三 seed 最坏：sequence exact=`0.001354`、slot=`0.701370`、ASK/repair/positive=`1/1/1`、negative false-allow=`0`、相对 predictive entropy drop=`0.198540`（正确 train-only baseline 后通过 25% 熵门）。这只修正了熵实验语义并改善安全头，完整组合仍未学会；没有模型选槽→真实 wire→fresh typed WebGoat GET/POST，promotion/training/memory/payload/vulnerability 全关。checkpoint SHA：37301=`401dd3a593209a26ab261c3cab7bef561613e2702b8036c23e2d3223bbad1ded`、37302=`ad26077b40ca76393459635b6c4bff1187546576a47af14608f8b720735d62c7`、37303=`b2a60dd46bd75f9e2d203aaec37c186c7a4973665414732942495891457b1ef6`。
- PG-373 已接入研究台只读 projection：`app/research_ops.py` SHA=`01d1821289d8e675bdcc732906c86157febc1c3846c5ae3e6a0aa2b6ca644a3c`，专项测试=`tests/test_pg373_research_ops_projection.py` SHA=`f8c180f004e88a31511490d5b0b2f500e9c0f1eb2d697d29e6c1a6eb1ec97581`。研究台只展示 bounded metrics/seed/checkpoint hash/locks，不显示 checkpoint path、token、wire 或 evaluator 内容。当前 rules SHA=`2400bb9bbfb3823bb0ec8b090b3da2b544a3e01d45b0a29342d20ba55ae2d6cc`。

### 快速框架规则的当前解释

- `framework_first_fast_lane_v1` 允许先搭建大框架、把不改变下一动作的细节延后；但 `authorization/reset/context firewall/split/capacity/negative/evidence` 永远不能跳过。跳过的字段必须显式 `unknown/incomplete/ASK`，不得用默认 false/absent 或平均分掩盖。
- 持久化 evaluator 状态只允许固定 digest、授权本地 network-none/loopback 的 disposable lane；每 seed/route/role 必须 fresh reset、数据库干净、前后状态 attestation、teardown 和 typed evidence。状态变化不是自动训练 gold。
- 训练快路径只在硬门已锁定后减少重复低价值检查；PG-373 证明“动作要快”不能替代正确 train-only baseline，也不能因为 A800 空闲就跳过词表 gap、熵门或模型选槽 live replay。

### 2026-08-09：PG-374 模型选槽→第二实现回放计划

- planning-only 合同已由代理完成：`scripts/plan_pg374_model_selected_replay.py` SHA=`7183d9cb62b5029ec2f69dd01d99ba5b83e37902ef00d49e460af3f3a729f966`，tests SHA=`cbbe4f9bc48c3ebcac775e7d732e8df4e5731ac3a01a3b350b7420bcf87858de`，plan=`research/pg374_model_selected_replay_plan_v1.json` SHA=`9f85d224ba065471327be14c595f14d71cb06da47c2ffef6c5a6ec4e4f7f6c10`，内部 `report_sha256=756ac7726329513e5c726524a15ff2d601eb9cd510d65179f52e0d90291bffc6`。
- 计划为 3 seeds×GET/POST×candidate/reference/negative/replay=24 rows；当前 `model_selected=0`、`typed_effect_confirmed=0`、`wire_created=0`、`target_contacted=0`。完整 13-slot 输入只有在下一次真实解码时才可标记 model_selected；缺 typed evidence 必须 `blocked_missing_typed_evidence`，`safe_to_send=false`，不生成 wire。未启动 Docker/GPU/网络，promotion/training/memory/payload/vulnerability 全关。
- 当前 rules SHA=`a5a1aeb3e12e8d19b21e315328cd90f1cbc5446d4d9721118f6d0d821dc61763`；AGENTS 自身哈希不写回备忘录。

### PG-374 研究台接入补充

- PG-374 planning artifact 已接入只读研究台：`app/research_ops.py` 当前 SHA=`668c058ff4ddfbd620bc494f8f119495370dc299cbde9d605f46058d77d69832`；PG-373 projection 测试 SHA=`f8c180f004e88a31511490d5b0b2f500e9c0f1eb2d697d29e6c1a6eb1ec97581`，PG-374 projection 测试 SHA=`9b10eab4b640142502967d35d3fbc47e613dc18a43063ce697c0be7379e19ddc`。研究台只显示 24 role rows 的计数和阻断状态，不输出 rows/route hash/wire/payload/响应；focused projection + PG-373/374 回归=`15 passed`。
- 当前关键实验 suite=`45 passed`，研究台+PG-373/374 suite=`67 passed`；一次全量 `python -m pytest -q` 在 244 秒超时，收集到 1576 项但未完成，不能宣称全量通过。后续若需全量验收应分组运行并为长时 Docker/浏览器测试单独设超时。

### 2026-08-09：PG-375 严格数据合同、组合解码器与 A800 表示预训练

- 严格过滤器=`scripts/build_pg375_strict_dataset.py`（SHA=`6154905d13f0f3373049c260ab31d97f5887f6acef4c803d480250e24aa1a775`），只读审计器=`scripts/audit_pg375_strict_dataset.py`（SHA=`afd8358bcfac742685550015a784ba755799bac734c2620e49874f71d3dc124f`）。数据=`research/pg375_strict_filtered_rule_ir_dataset_v1.json`（SHA=`3c40f0841ab1d8f6d22030f3ca3400b68f158175ea5a7c9267af6fa544210c72`），审计=`research/pg375_strict_filtered_rule_ir_audit_v1.json`（SHA=`b5c8c9333c2dc09e33461824de0ab5b1ee0aca39742564658850b59e53edc5b6`）。原始 2600 行经 holdout-precedence 去重后 active train=`1208`、active holdout=`572`；排除 train=`392`、隔离 holdout=`428`；active 跨 split context/exact overlap=`0`、unknown context/target/slot=`0`、raw hits=`0`。这些数字只表示 candidate audit 通过，不表示 capability training 授权；输出明确 `training_eligible_rows=0`、`capability_training_allowed=false`，隔离项仅保存哈希引用与原因。
- 组合 runner=`scripts/run_pg375_composed_rule_ir_candidate.py`（SHA=`1295550ba67c518509741e7fea7f29cc6dbfc73d516d7e6c64656234160e95e4`），tests=`tests/test_pg375_composed_rule_ir_candidate.py`（SHA=`fe5c706c4f2dece7cad72a81d89ebdb7731990f443bc9d2c65944c5726dc6841`）。架构是共享 CausalMoE、13-slot 自回归 Rule-IR composition decoder、next-token/slot/ASK/repair/negative 辅助头，Stage-A/B 均 target-only mask。严格 source contract 缺 operator review、typed evaluator 和 fresh role reset，因此计划报告 `research/pg375_strict_candidate_plan_v1.json`（SHA=`f96dca9d3138cc7b55c91be15f7c75d3a7987ad46ca27863bb22366e12f44a02`）在 optimizer 前 `blocked_data_contract`；没有 GPU、Docker、网络或 checkpoint 运行，promotion/memory/payload/vulnerability 全关。
- 单独的 context-only 表示预训练 runner=`scripts/run_pg375_context_representation_candidate.py`（SHA=`6320886c1f076b9594287b8dbe99c7842c0afa3bfcb42bc8e916ac1b056d7013`），tests SHA=`194b62d9f8ae61c927683b025212a4f1f5ce531568ecc3f14082b4474de7235b`。它只读取抽象 `context_tokens`，train-only vocab=`738`（含 PAD/UNK 总=`740`），窗口=`606`，holdout unknown=`0`、overlap=`0`，不读取 target/raw/evaluator。周末在 `112.111.7.91:60228` 的 A800 GPU0、`CUDA_VISIBLE_DEVICES=0` 上跑 3 seeds=`37521/37522/37523`、4 epochs、batch=`16`；GPU1–7 未触碰，Docker/network=`false`。report=`artifacts/pg375-context-representation-a800/pg375_context_representation_a800_v1.json` 文件 SHA=`d2dae776e15ac57a12f8bd96ac328baef462000139d9dc0051978c1e2a7247ef`，内部 SHA=`244ce9872ddb4b6d05072ec87177814347b2ecfafe94412ccd808f065b4cc86b`；最坏 holdout token accuracy=`0.909599`、predictive entropy=`0.815927 nats`。这是表示学习 candidate evidence，不是漏洞检测、Rule-IR 组合、模型选槽或 payload 生成，所有晋级标记继续关闭。
- 结论/下一步：模型现在有可复核的整页 context 表示预训练结果，但 capability 模型仍未通过 source-grounded typed GET/POST candidate/reference/negative/replay。下一轮必须补第二独立动态实现的真实字段与 `syntax_category_ref/payload_shape_ref`，再用同一 13-slot decoder 做低学习率 SFT/受限 RL，最后接 model-selected binder→fresh typed replay；不能把 A800 表示 loss、离线 slot 分数或单一 synthetic evaluator 结果冒充“会发 payload”。

### 2026-08-09：PG-368 完整 WebGoat fresh method-shape 回放

- 在固定 WebGoat digest、`PG368_LOCAL_DOCKER_EVAL=1`、network-none、loopback relay、无端口/挂载条件下完成三 seed×GET/POST×candidate/reference/negative/replay：`research/pg368_webgoat_binder_replay_full_v1.json` 文件 SHA=`1acd44054c05a56ce530fadae1931e528b7acbcd70549d5f6f7121c652206cea`，内部 `report_sha256=86069c06d76c06bb8fb740c11fdde64e43f68e63ce5f0e48bd537412b40e4886`。24 个 role fresh 容器均联系目标，typed method-shape=`18/24`，negative violation=`0`，model_selected=`0`，ASK=`24`，confirmed_positive=`0`；容器已销毁。
- 这只是 evaluator-only method-shape 证据，不是漏洞复现：当前 runner 没有 PG331 七轴 `field_capture_manifest`、role reset/target digest、failure→repair/belief/replay 等完整 source-row 字段，不能把 24 行转成训练样本或长期记忆。结构形状不等于漏洞存在，模型仍没有从 Rule-IR 选槽到 wire 的能力证据。

### 2026-08-09：PG-376 高容量 context-only A800 反例

- PG-376 runner=`scripts/run_pg376_highcap_context_pretrain.py`，本地 SHA=`66bec0c0d42e4b1e77ca6e56acd8a491c74c29bd5607f170191031437f5639b9`；远端实际锁定副本 SHA=`de27de35184e5f06c965df36fd5d5a75cdd592e4eeafbc3bb16b2b04c35856dd`，已保存为 `artifacts/pg376-highcap-context-a800/pg376_remote_runner_de27.py`。tests SHA=`89d7af102a425ebaae041c5686c1e05414d3aa7be1a95a227a5e8cb08715010e`，focused PG375+PG376=`8 passed`。
- 周末远程 `112.111.7.91:60228` A800 GPU0、`CUDA_VISIBLE_DEVICES=0` 跑 3 seeds=`37601/37602/37603`、4 epochs、batch=`16`、d_model=`512`、8 layers、8 experts、expert hidden=`2048`；本次只触碰 GPU0，GPU1–7 未由本实验触碰，Docker/network=`false`。报告=`artifacts/pg376-highcap-context-a800/pg376_highcap_context_pretrain_v1.json` 文件 SHA=`62749c2e1208ccb339572455bbf20fe0b444ce6645c3ae57ed40deb8a891e2a6`，内部 SHA=`d7643ae9a10858b9ea6c350aeefa935344f2ddf1193c109b9fcb9b136f07a91a`。
- 结果：最坏 holdout token accuracy=`0.943419`，但 predictive entropy 相对下降=`0.938598–0.945535`，远超 25% 信息保真门；`target_tokens_read=false`，因此这是 context 表示上限的失败对照，不是 Rule-IR、ASK/repair、漏洞检测或 payload 能力。seed-37601 本地 checkpoint SHA=`0160a3bb1bada7da85721589af0a43735f0c4d45650a4873afa5a68cfb7fda2c`；seed-37602/37603 仅保留远端 SHA=`ed5ac8ed7c4c0dd972e56357fe7deb0e617e86548897799082d45b31d01c9440`/`9b2d8f97755d4d464de9f275fb80d5ade3ad6d46ec8958b5b0254149a9872c76`。
- 研究结论：扩大容量让表面 token accuracy 变好，却让分布熵更快塌缩；今后不能以“更大模型/更多 epoch”解决问题。下一动作回到 source-grounded 七轴、`syntax_category_ref/payload_shape_ref`、失败改动作和跨实现 typed replay，再用真实 Stage-A baseline 约束 capability SFT/RL。

### 2026-08-09：PG-377 熵保真 context 对照与 WebGoat source-row 适配器

- PG-377 runner=`scripts/run_pg377_entropy_preserved_context_candidate.py` SHA=`af3fe6017db8ec24b3ee873d4e4f103b1080ce3af4785b3560b3cc2aa9cd7020`，tests SHA=`fd0e73f2c772f7e90781c52fdb7544944631dc63a5b0a6e312d36bc50d3bc4e8`。它只读 PG-375 strict active train context，词表为 train-only（738+PAD/UNK=740），holdout 不参与优化，teacher 是已训练的 PG-375 context checkpoint=`6c7c54056d6a0d4ba1fcef9a4e8fd281d79bc1e798e41730f4eaae107eadaecc`；学生使用 KL=1.0、temperature=2.0、entropy weight=0.25，不能访问 target/evaluator/raw。
- 周末远程 `112.111.7.91:60228` A800 GPU0、`CUDA_VISIBLE_DEVICES=0` 完成 3 seeds=`37701/37702/37703`、4 epochs、batch=`16`、d_model=`512`、8 layers、8 experts、hidden=`2048`；GPU1–7 未由本实验触碰，Docker/network=`false`。报告=`artifacts/pg377-entropy-preserved-context-a800/pg377_entropy_preserved_context_candidate_v1.json` 文件 SHA=`688430d9748b58ac19f2f4680855f16925da9bc92c8b56ed73cf97cd217c7a73`，内部 `report_sha256=9ea58595b571db0ca84475313a4329254ba201f5bdab9aa49292867e753d6de3`；训练时 rules lock=`6beef2d0358c925b5b926542fe2a38487bb462909819f32bec244f929e1c8a19`，登记本轮后 rules SHA=`7c65ba31c55741748f2edcf0a7efa167ee8ad6203df28280c8a0e8878099bfbd`。
- 结果：teacher holdout entropy=`0.790052` nats；学生三 seed entropy=`0.548009/0.503714/0.526345`，相对下降=`0.306363/0.362429/0.333784`，最坏=`0.362429`，超过 `0.25` 信息门；token accuracy 最坏约=`0.942485`。KL/熵匹配没有解决容量造成的分布塌缩，状态=`blocked_entropy_preservation`，不是 Rule-IR、漏洞检测或 payload 能力，training/memory/payload/vulnerability 全关。远端 checkpoint 只登记哈希：37701=`bc6f1701a0a2e9abbb34aead5fe0b7ac0118e8f0c640cbfa39620456dee81837`、37702=`b23c04c56882ad9d7df1c26d7a980d14e4a7d318869fec6f90cacb46323d22e1`、37703=`f48df1100c67c535a80c05b18c4327ebaa4596bc64d5b45e9680e5851c1351b5`。
- 新增纯内存 WebGoat 适配器=`app/pg377_webgoat_source_row_adapter.py` SHA=`f46750b8831e985bf04e19c3c8e218d782c1bb473230468218e6e9a6033a288b`，tests=`tests/test_pg377_webgoat_source_row_adapter.py` SHA=`8ac6ef08007b354c5ab4fa1d7da4c6aec989aeffb76bb5728b4ab2cdd614d09c`，focused=`10 passed`。适配器输出 PG-331 七轴与 107-field manifest，GET/POST 缺观测→ASK，失败同动作→repair/observe，拒绝 raw URL/payload/response/wire；evaluator sidecar 只在 off-context。它没有启动 Docker/网络，也不产生 training-eligible source rows，不能把 PG-368 method-shape 证据伪装成漏洞样本。

### 2026-08-09：PG-378 teacher-residual context candidate

- PG-378 runner=`scripts/run_pg378_teacher_residual_context_candidate.py` SHA=`a12dded68e238014482960909449d69f5ee5a19687884d4d5ea5813d779331cc`，tests SHA=`403148df5e14ac913dc342f6b47bca0c635ca53603ee2b0f69c52f13b7d42946`；仅读 PG-375 strict abstract context，train-only vocab=738+PAD/UNK=740，holdout=572 不参与优化，target/evaluator/raw 均未读取。
- 周末远程 `112.111.7.91:60228` A800 GPU0、`CUDA_VISIBLE_DEVICES=0` 完成 seeds=`37801/37802/37803`、4 epochs、batch=16、d_model=512、8 layers、8 experts、hidden=2048、residual_scale=0.1、KL=1.0；GPU1–7 未触碰，Docker/network=false。报告=`artifacts/pg378-teacher-residual-context-a800/pg378_teacher_residual_context_candidate_v1.json` 文件 SHA=`10baE3BD252A21D5DEC4E6AF6DE17C8E1FABD0C9D4CC9370402A9799723D5131`，内部 `report_sha256=94d708d689b0b77795b9724bb350535ac00a2b3f5dc805fa111f950619acfab2`。
- teacher holdout 熵=`0.790052`；三 seed 学生熵相对绝对变化=`2.9864%/2.8836%/2.5502%`，worst=`2.9864%`，通过 `≤25%` 熵保真门；token accuracy=`0.913192/0.913305/0.915370`。这是表示层的 function-preserving 候选证据，不是 Rule-IR、漏洞检测或 payload 能力；`capability_training=false`，training/memory/payload/vulnerability promotion 全关闭。checkpoint 只登记 SHA：`468d2360465870fff41d4f7605ebc274c7623ee68313748ca0c071b4e4c90a27`、`54c6b6910077f9ba4b00d89838ff9d284758ac3e1b4b7b884596401ca535cc52`、`9ffa58a0e5cfbfabe7c27c4561471919395d460ac7073a21360495bfa09a424f`。
- PG-378 训练锁使用 rules SHA=`7c65ba31c55741748f2edcf0a7efa167ee8ad6203df28280c8a0e8878099bfbd`；登记 PG-378 结果后当前 `research/improvement_rules.json` SHA=`70bb048127f6214aa1e8f5ab876921e351c0d74a69d76e3706eaf3d399d20e7e`。下一步仍是完成 PG-377 fresh WebGoat 七轴 source-row 采集与独立审计，再谈 capability SFT/RL；不能把 PG-378 表示熵门通过当作漏洞能力。

### 2026-08-09：PG-377 WebGoat fresh 七轴 source-row 采集

- 修复首轮工程失败：第一次 live run 因 relay readiness 的诊断字段误传给严格 adapter，未产出样本；该失败只作工程诊断。第二次使用 `_normalize_reset` 后完成 3 seeds=`37701/37702/37703` × GET/POST × candidate/reference/negative/replay，共 24 个 fresh role rows，容器逐 role 清理。
- 当前 runner=`scripts/run_pg377_webgoat_source_rows_live.py` SHA=`e587bf11923da7ea47c94cfebb812cb2cbfc5c3959c8d242801fc727aa51e289`，tests SHA=`0414de384b6d7e22967bfb08b0c56c2c9d1e355ff1649c87fb6bab753b7a5983`。报告=`research/pg377_webgoat_source_rows_live_report_v2.json` SHA=`502c4b08d4bff3d7755bc6b4225d7b994c8e6dcc3a427c3573ecad46f9081ee3`，内部 `report_sha256=b7fc3423f71f5987944ef30645489957c29b9faed5638d88c2e1981270d33031`；dataset=`research/pg377_webgoat_source_rows_live_v2.json` SHA=`35dff4d634c83efbe81d7fae559cbd1895ba17a64f483e1523cd528ebc249620`；sidecars SHA=`62edf34c062117cdb94f756fb5cb7ef5df12b7928568cf09d5803da39aef135f`。
- 采集结果：24/24 source rows 严格有效、七轴与 107-field manifest=24/24、typed role=18、negative violation=0、failure observed/action changed=6/6、belief/replay=24/24、strict incomplete=0、context forbidden token=0；但所有行都是 `implementation_holdout`，`operator_reviewed=false`、`training_eligible=0`，所以不能直接训练或晋级。报告 `status=completed_source_row_candidate_only`，所有 promotion flags=false。`execution.network_contacted=true` 仅表示 evaluator 通过容器内 loopback relay 发起了受控请求；source attestation 仍是 `network_mode=none`、`loopback_only=true`、无端口/挂载，不能解释为外网访问。
- 独立 PG-331 source-row audit=`blocked`，audit hash=`d0b31d820a8e01041c326efbc4c3227eca83da16398b6591df1cbe171dc556b`；唯一硬失败为 `empty:training_eligible_rows`，unique sequence ratio=`0.333333`，split isolation clean。该数据证明“整页采集合同可运行”，不证明 WebGoat 漏洞、payload 或通用网址能力；下一步必须增加独立实现/多页面训练 split，再做 audit 后才可 capability SFT/RL。
- 登记 PG-377 live 与 PG-378 结果后当前 `research/improvement_rules.json` SHA=`da35f937cca3570ea497181f3ab33569dd16f927e398d5e8291dba84439fb287`。
- 下一动作不是再加容量：先用该适配器接入真实、授权、fresh role-level WebGoat 页面采集，补齐 reset/target digest、candidate/reference/negative/replay typed evidence、failure→repair/belief 与完整字段状态；之后重新做 source/implementation holdout 审计，再申请组合 Rule-IR SFT/RL 和模型选槽→binder fresh GET/POST 回放。

### 2026-08-09：PG-385 抽象对抗快速展示规则与 PG-379 双实现动态候选

- 规则新增 `pg385_abstract_adversarial_evaluator_fast_lane`，登记在 `research/improvement_rules.json`（当前 SHA=`3c28fef77f3d680edd4339ca26e55d8376f87c3839773019f56714f49aaef43d`）。允许独立的抽象 WAF/filter、编码链、语法类别、payload-shape、失败→单变量修复、negative/replay、stateful evaluator 差分数据；模型只输出抽象 Rule-IR/ASK/allowlisted variant/ref，不输出任意原始字节、原始响应、evaluator answer、任意外部目标或回调。具体字节只能由 source-attested、reviewed、固定 loopback evaluator 最后一跳临时绑定；stateful stored challenge 也必须 fresh container、reset before/after、DB clean、teardown，状态与 oracle 永不进 context。首阶段 next-token predictive entropy 仍是 `<=25%` 硬门；后续压缩/adapter 熵仅诊断，但 finite logits、非空支持、无泄漏、slot/ASK/repair/negative/fresh/typed/context firewall 仍是硬门；promotion/training/memory/payload/vulnerability 全关。规则测试 focused=`4 passed, 5 deselected`。
- PG-384 可作为明早的安全展示基线：抽象组合 holdout 48 行中 6 行模型选槽、42 行 ASK、unsafe allow=0；本机 synthetic loopback evaluator 中 typed/replay/negative clean 均 `6/6`，但它不是任意网址漏洞、通用 WAF 绕过、持久化 XSS、反链或可迁移 payload 生成。
- PG-379 双动态实现 live candidate 已完成 3 seeds×12 route classes（GET6/POST6）×candidate/reference/negative/replay：role episodes=`288`、source rows=`216`、adapter-valid=`216`、typed=`216`、negative violation=`0`、failure/action-change=`72/72`、capture failure=`0`。报告=`artifacts/pg379/pg379_dynamic_source_rows_live_report_v2.json` 文件 SHA=`89f3576a9280067afb790e47a8a8dc61e24858dddc455d79022e5fc2668bf321`，内部 SHA=`354b44714efa18a55dade518fb63fb83d2d214d51175973b3dc392586f2aca53`；sidecars SHA=`bb460bd8e4b962754a5887c567936a62bafc0f2477ccc57e1995957f79692a94`；rows SHA=`0e1a59f94f8a6a9a910cf2e206ef1c7d22cdc33d2ff041aae9e9b318c2a52350`。实现 A image digest=`sha256:7c1748697d83219cc876eb5f6c199211e0add7e943ac9d971e030d52d97a1471`，source SHA=`c9712d77767f0f37ee55a986ed7f1562162c509e4e9c358be74db1cc783245d9`；实现 B image digest=`sha256:efb5a40d408159de431368e99cdece31c474d43a6b7878e4e348de1fc4ddfaf0`，source SHA=`20a69582a0be68f4086cbb6f5462b0b8628b93c0fbf07de9e4abb790280d4402`。均为 `network=none`、loopback-only、无外部网络/GPU/训练，所有 promotion false。
- PG-379 严格 PG-331 validator 只接受 `27/216`，另 `189` 行为 explicit incomplete（主要字段状态仍 unknown），strict training-eligible=`0`；因此不能用“adapter-valid 216/216”冒充完整训练源。当前最重要的工程修复是 POST JSON state-transition route `a5d4366e78a5323351a2e9ce26aa238a47593a2c40a8b903ddb11bb09f1c4acc`，定向 fresh replay report/sidecars/rows 分别为 `artifacts/pg379/pg379_post_json_state_transition_report_v2.json` SHA=`dc7821707a7074f01cbf959de5955e70c6ff251691197f16fdd7cc2aa0fa61c1`、`...sidecars_v2.json` SHA=`bc36b0f89982856769cfcb0e0df9c1a7e07d72a45af7f5f0f99c4432b586bd21`、`...rows_v2.json` SHA=`2e058d07d8e3b26a0d662efa8b056e7c0ae07b61f0e59f64f4f6e73fa948e6e1`，typed=`18/18`、negative=`0`、failure/action=`6/6`。这只修复了 evaluator 工程/观测，不等于 payload 能力。
- PG-379 collector=`scripts/run_pg379_dynamic_source_rows_live.py` SHA=`b85c5fd9258f61fe9d057441540589c542a380ee57f2ee706b9468251ce9fd87`，Docker wrapper=`scripts/run_pg379_docker_source_rows_live.py` SHA=`ad7cfba71dc36478ea68587d5ba8d255d0425078af509dcfa79e824a5995bbd7`；当前没有运行中的靶场或训练进程。展示只能说“抽象条件→受审阅本地模板→typed effect/repair”，不能说“模型已经学会对任意网址生成 WAF 绕过/持久化 XSS/反链 payload”。

### 2026-08-09：PG-385 canary 发送语义更正

- 用户明确的展示范围是“给定本地过滤器/动态页面中的有限逻辑例子”，不是任意目标攻击器。因此规则现在允许 `benign_canary_wire_allowed=true`：模型仍只输出抽象 Rule-IR/variant reference，reviewed evaluator 可在最后一跳绑定一个非外连、非凭据、非 timing、非破坏性、非业务写入的 canary marker，并在固定 loopback 页面实际发送一次；可持久化保存的仍只有槽位、模板/运行时哈希、请求哈希、有限响应投影和 evidence SHA。`model_can_emit_raw_canary=false`、`arbitrary_target_or_external_callback=false` 不变。
- 这个“能发测试 payload”只适用于预注册的本地 fixture：过滤器拒绝、反射/形状差分、状态差分、失败→一次变量修复、负对照和 fresh replay 都必须有 typed 结果；未绑定模板、缺字段、外部 URL、反链、凭据、真实持久化数据或不可复现状态一律 ASK/abstain。PG-385 规则最终 SHA=`504fee444baadf61467906dc1b2fae7910240b792fb6e1a908d91345375db06a`。
- 因此明早可交付的是“模型选抽象变体→本地 evaluator 绑定安全 canary→发送→脱敏 typed 判定→失败修复/复放”的展示闭环；不能把 canary 例子包装成通用 WAF 绕过、持久化 XSS、反链或任意原始攻击字符串能力。没有新增远程训练或 GPU 占用。

### 2026-08-09：PG-384 最后一跳 canary wire 展示

- `scripts/run_pg384_model_selected_binder_replay.py` 新增显式 `--show-wire`；仅在 `--live` 且 `PG384_LOCAL_EVAL=1` 时把最多 3 条本机 loopback canary wire 打到 stdout，`report`/checkpoint 仍不保存 wire、URL、body 或 canary。脚本 SHA=`b28bb2fac56b551c0a1b8f7290cd19fb8bd12332d19f8213b56c585141047a22`。
- 已实际运行 `--live --show-wire --max-rows 48`：holdout=`48`、decoded exact=`48`、model-selected=`6`、confirmed typed=`6`、ASK=`42`、unsafe allow=`0`、binder reject=`0`；临时展示报告=`research/pg384_model_selected_binder_wire_demo_v1.json` SHA=`fde54fe1683288259a9b20914ca6db25287ce5926dd62b97dc9b6594e7e1fe51`。stdout 仅展示形如 `GET http://127.0.0.1:<ephemeral-port>/...?...=PG384M....CAND` 的无害 marker wire，运行后 synthetic server 已关闭。
- 这证明“模型选槽→模板最后一跳生成具体测试字符串→发送→typed candidate/reference/negative/replay”已经可演示；它不证明任意原始攻击字符串、通用 WAF 绕过、外部反链或持久化 XSS。后续若增加形状/编码，只能扩展预注册本地模板并保持同一 evaluator/negative/fresh/证据门。

### 2026-08-09：PG-385 过滤反馈→编码修复→canary 复放

- 新增纯本地 fixture=`app/pg385_filter_canary_fixture.py`、runner=`scripts/run_pg385_filter_repair_demo.py`、tests=`tests/test_pg385_filter_repair_demo.py`。fixture 将单层分隔符 canary 标记为 `filtered/encoding_filter/raw_delimiter_blocked`，只接受抽象 reasoner 选择的第二编码层；不执行脚本、不写业务状态、不访问外网、不返回提交值。
- 已实际运行 demo（`python scripts/run_pg385_filter_repair_demo.py --show-wire`）：baseline filtered=`1`，model repair selected=`1`，action changed=`1`，candidate/reference/replay typed=`1/1/1`，negative violation=`0`。报告=`research/pg385_filter_repair_demo_v1.json` SHA=`bedb8810fc2f5f769766e74099c87f2835b207c8d4bbb3c379159d9fbd5c3691`；测试=`3 passed`。stdout 的具体 wire 只在本机临时显示，报告只保留抽象反馈、槽位、wire/evidence hash。
- 该证据正好对应“先发被过滤的测试字符串→模型读取脱敏失败反馈→选择一变量编码修复→evaluator 生成下一条测试字符串→typed effect/negative/replay”。规则登记后最终 `research/improvement_rules.json` SHA=`86a5aaa7af3dc4ec81f0ac7497df2bf80ab7730a1fe1346edffcc84bd6076c18`；仍 `training_eligible=0`、promotion 全关。

### 2026-08-09：PG-385 独立抽象过滤修复数据集

- 数据构建器=`scripts/build_pg385_filter_repair_dataset.py` SHA=`855575341d3034e2fc86875c5d5b2b0b36fc59b517339462d324b453e4f6945c`；数据=`research/pg385_filter_repair_adversarial_dataset_v1.json` SHA=`273ae959ef133aeba9dfed723321a36aa7125b84d35bf8d23410a9c245077a9b`；测试=`tests/test_pg385_filter_repair_dataset.py` SHA=`9717d843f58c11e387f8f09998c8e62644c4c678d5ece99d7429138d4810e7d3`，focused=`5 passed`。
- 数据集共 `128` 条抽象记录：`train=64`、`implementation_holdout=64`、2 seeds、GET/POST、4 场景（编码规范化、分隔符/语法门、形状/长度门、解析边界恢复）、candidate/reference/negative/replay。每条只保存过滤状态、失败签名、编码/语法/形状槽位、ASK/repair/negative/replay 目标；原始 canary、URL、wire、响应体和 evaluator answer 均不入数据，`training_eligible=0`。
- 规则登记该数据集后最终 SHA=`d49eddb0db7a45842f5400ab09b7027c6535a07f734909d155b05487ccdb42ac`。它可作为抽象修复 SFT/离线候选材料，但仍必须经过独立实现、fresh reset、typed C/R/N/replay 和第一阶段熵硬门；不能把抽象 repair 标签直接称为通用绕过能力。

### 2026-08-09：PG-385 抽象数据 CPU wiring smoke

- 复用 `scripts/run_pg380_abstract_reasoning_sft.py` 对 PG-385 128 条抽象记录做受限 CPU smoke（row_limit=32、d_model=32、1 layer、1 epoch），报告=`research/pg385_filter_repair_sft_cpu_smoke_v1.json` SHA=`7bba729c604622e6b9fb09c5b6e7cc0dcfdc64c38676f5ae586d5805c9b89043`。
- 结果是候选 wiring 诊断而非能力成绩：sequence exact=`0`、slot accuracy=`0.365385`、ASK=`0`、repair=`1.0`、positive=`0.875`、negative false allow=`8`、entropy drop≈`0.000201`。因此 capability gate 失败，未提交 A800；这说明当前抽象数据/目标可以喂入 token-MoE，但真实模型还没有学会安全的 ASK/negative 组合，不能包装成“模型已会绕过”。
- 登记 smoke 后最终规则 SHA=`adf47b34b372e6056ce4fba66032d6335d3f383e03c263a400ad10ce26f1b0b4`。下一步应先修 train-only target/negative/ASK 组合与跨实现留出，再申请候选训练；不以占用 A800 替代失败的能力门。
- 规则哈希更正：上方 PG-378 段落记录的是其训练时锁 `7c65...` 与登记前快照 `70bb...`；在 PG-377 live 结果登记后，当前有效 `research/improvement_rules.json` SHA 为 `da35f937cca3570ea497181f3ab33569dd16f927e398d5e8291dba84439fb287`。

### 2026-08-09：PG-333 capability SFT 兼容性审计与 PG-379 采集矩阵

- PG-333 三实现 merged rows 的只读审计新增：`scripts/audit_pg333_capability_sft_compat.py` SHA=`12a590feee02dc7813c76ef4804f21a16bda61207aa09b4943cadeaf5d98830`，tests SHA=`d0972784cfdcf5ccd0005a7af1a4466985c87b23d66e671df3eea99880e231`，报告=`research/pg333_capability_sft_compatibility_audit_v1.json` SHA=`5c87302b04bf4b0e467dc54f1926585f9a52fb472395a5f607031ca5ce75a4ba`。45 行（train=9、implementation_holdout=36）表面上有 30 条旧 `training_eligible`，但严格 accepted=0；13-slot 目标缺 `ask_reason/syntax_category_ref/payload_shape_ref/oracle_ref/negative_control_presence_ref`，目标只有 10 tokens/8 fields，holdout train-only vocab unknown=159，required window=4145。PG-333 merged information/capacity gate 仍 diagnostic/blocked，不能提交 A800 或 capability SFT。
- PG-379 planning-only：`scripts/plan_pg379_source_collection.py` 当前 SHA=`4af527d2e215988337bfb5e3a0819b2deefc5a181e9e87910c613c42ed0ea0da`，tests SHA=`5b49708b563ffb86a457ceac7cb4040f0541d0484ae568cc11d92a2e084bb85a`，计划=`research/pg379_source_collection_matrix_plan_v1.json` 文件 SHA=`19a1822e8b762da0e2598fbbd2e4c3ad8a00b8fd10a2792ead2c12a06a72881d`，内部计划 SHA=`b510e350d8d0dcede8b206c1fe7c2e128aabe6bf9201d6450a8fb085691a5458`。它保留旧 split，目标两套独立动态实现、每套 12 route classes（GET6/POST6）、3 seeds、candidate/reference/negative/replay，计划 source rows=216、role episodes=288；当前不启动 Docker/网络/GPU、不生成训练行。

### 2026-08-09：PG-378 residual-scale=0.2 A800 熵边界消融

- 周末远程 A800 GPU0、`CUDA_VISIBLE_DEVICES=0` 完成 3 seeds=`37801/37802/37803`、4 epochs、batch=16、d_model=512、8 layers、8 experts、hidden=2048、residual_scale=0.2、KL=1.0；GPU1–7 未触碰，Docker/network=false。报告=`artifacts/pg378-teacher-residual-context-scale02-a800/pg378_teacher_residual_context_scale02_v1.json` 文件 SHA=`a99fd31ed4e79b000c6b2cbce74b0cd3f907cc10f971bb551c649aa1776138d8`，内部 `report_sha256=8b4c22ec2008e5f4df11f3dd3ba972c1a2dbfbabd421104d08edd7fc6bf2b207`。
- teacher entropy=`0.790052`；学生 worst token accuracy=`0.914154`，熵相对绝对变化=`3.8829%/3.5010%/3.3783%`，worst=`3.8829%`，通过 `≤25%` 保真门。状态=`residual_entropy_preserved_candidate_only`，仅表示层边界证据；target/evaluator/raw 未读，capability/training/memory/payload/vulnerability promotion 全关闭。A800 已在完成后释放。
- 登记本轮结果后当前 `research/improvement_rules.json` SHA=`08a3ce980e20f938bd7f36c53128c3d2770ad257c793b4ce57440de66e8943cd`。下一次 GPU 占用必须等待 PG-379 真实 source/implementation split 与完整 13-slot 审计，不再重复无信息增益的容量 smoke。
- 规则哈希最终更正：PG-333 audit、PG-379 plan 与 PG-378 scale02 结果均已登记；当前有效 rules SHA 为 `08a3ce980e20f938bd7f36c53128c3d2770ad257c793b4ce57440de66e8943cd`。

### 2026-08-09：PG-380 抽象对抗推理数据与分层熵规则

- 新增独立的抽象对抗推理数据集：`scripts/build_pg380_abstract_adversarial_dataset.py` SHA=`15d498f1df5247fb69d6fa10c1b1d96d4fb2ace13dc7d1ad6b9e3612d7077268`，数据=`research/pg380_abstract_adversarial_reasoning_dataset_v1.json` SHA=`9bcc7a20c99b8af171ef86b3f263f026beafa6afafa2fd66a77c7d0eab64ef00`，内部数据 SHA=`168f942939f8d49fa07e65e36a62de9077865aea704b542c388fe89a22c55ab0`。3168 行（train=2112、implementation_holdout=1056、2 implementations、12 surfaces、GET/POST、candidate/reference/negative/replay），上下文/目标只含抽象 surface、参数角色、编码链、语法/形状、过滤反馈、ASK、repair、negative、replay Rule-IR；raw payload/响应体/URL/wire/evaluator 答案均不入模。
- 纯抽象 reasoner=`app/pg380_abstract_probe_reasoner.py` SHA=`47b87d1ccca152ab9cc680f48c53f8fbb2684270de0c15c8dba2f33e9d121aa5`；测试=`tests/test_pg380_abstract_adversarial_dataset.py` SHA=`2c662ce83fc27658500f21a198a48c86beca47ccb8caed3f0355f4b14ede5441`、`tests/test_pg380_abstract_probe_reasoner.py` SHA=`92483b98806a4ce5137f80a6734b56d3a5873b04b229bebf51a49cca3a5fad89`。reasoner 缺信息时 ASK，过滤反馈只做 one-variable 抽象修复，具体字节仍只能由 reviewed local evaluator template 最后一跳绑定。
- 抽象 SFT runner=`scripts/run_pg380_abstract_reasoning_sft.py` SHA=`3682e7fe44b8fc4e20173ceeb47c5740909afb04593c12a9bb1541c760a4be7a`；周末 A800 GPU0（`CUDA_VISIBLE_DEVICES=0`，3 seeds=`38001/38002/38003`，d_model=512、8 layers、8 experts、hidden=2048、4 epochs）报告=`artifacts/pg380-abstract-reasoning-sft/pg380_abstract_reasoning_sft_candidate_v1.json` SHA=`c6fc6da2abd685d02ae1930ae260268648cd130ed6de1a73f38f3cc5d8e52c21`。ASK/repair 最坏均=`1.0`、negative false allow=`0`，slot 最坏=`0.75947`、sequence exact=`0`；后续候选未学会完整 13-slot 组合，所有 capability/promotion 仍关闭。
- PG-380 的熵下降约 `0.970033` 仅作为分层策略变更后的后续层诊断，不得用来宣称能力：第一阶段 next-token 预训练仍执行 predictive-entropy 硬门；后续压缩/adapter 层不再以熵单项阻断，但必须报告有限 logits、非空类别支持、无 holdout 泄漏、slot 覆盖、ASK/repair/negative 门。该策略不会放宽授权、fresh reset、evidence、context firewall、split 或安全晋级门。
- 规则外部 SHA（本次 PG-380、PG-378 scale03 证据和分层熵策略登记后）=`b8e0a2a420bac664d50322c5623c45b9874629b4406fbab72228f55926245fda`。下一步只允许在已授权 loopback evaluator 中展示“抽象推理→reviewed template→typed oracle”的最后一跳；不把任意 WAF 绕过、持久化 XSS、反链或攻击字符串写入模型/长期记忆。

### 2026-08-09：PG-381/382/384 抽象组合与绑定候选（分层熵规则生效）

- PG-381 高容量组合 runner=`scripts/run_pg381_abstract_composition_candidate.py` SHA=`4bf9f90d894b9c6502843e28bd8526b3917f528ccb5b9e00c26faa48148de074`；在原 PG-380 组合留出上保留为失败对照：slot composition exact 最坏=`0`、slot accuracy 最坏约=`0.782`、ASK/repair=`1.0`、negative false-allow=`0`。未见 surface token 导致组合泛化失败，不用补标签或伪造泛化。
- PG-382 因子化抽象数据 builder=`scripts/build_pg382_factorized_adversarial_dataset.py` SHA=`fe00557077021406247d3fbeba2329431dcca516f044ad2a0caa63763ac9d1c2`，数据=`research/pg382_factorized_abstract_adversarial_dataset_v1.json` SHA=`d8fb4c69936a69fd498347468a88caa59603afcad6409c05a4f11f4744f66841`：6336 行（train/implementation_holdout 各3168、2 implementation、12 surface、GET/POST、4 role、24 source hashes），仅抽象因子和 sidecar，raw payload/response/evaluator 不入模。
- PG-384 绑定抽象数据=`research/pg384_binding_abstract_adversarial_dataset_v1.json` SHA=`bf000165d7498dde348db97452aa62a597cc54990860916754532e047e67e1fd`；使用同一高容量组合 runner 在周末远程 A800 GPU0、`CUDA_VISIBLE_DEVICES=0`、3 seeds=`38101/38102/38103`、GPU1–7 未触碰。报告=`research/pg384_binding_composition_candidate_v1.json` SHA=`627a01cbd9b4b8fbbbe741f61e8c81df7006b205f87eb38535023492a0bd70a9`；sequence/slot composition/slot accuracy/ASK/repair/positive 最坏均=`1.0`、negative false-allow=`0`、熵相对变化最大=`-1.637033`。
- 结果边界：PG-384 只证明抽象 Rule-IR 组合标签在因子化合成留出上可学习；`model_selected_wire_replay=false`、`typed_live_replay_with_model_selected_wire=false`，仍未证明模型能在第二独立实现上复现漏洞，也不提供任意 WAF 绕过、持久化 XSS、反链或攻击字符串。具体字节只由本地、授权、source-attested evaluator 模板最后一跳临时绑定，持久产物仅保存抽象槽位/哈希/有界响应投影。
- 本轮专项回归：PG-380 规则/数据/推理 + PG-381/382/384=`20 passed`（2 个既有 Transformer nested-tensor warning）。规则已明确：首阶段 next-token 预测保留 `≤25%` 熵硬门；后续压缩/adapter 层熵仅诊断，仍必须检查有限 logits、非空类别、无 holdout 泄漏、slot 覆盖和 ASK/repair/negative 安全门。当前 `research/improvement_rules.json` SHA=`1eb1467243e16149f01a9c56167a71bf2f547e972b30a89c699c489d2a121f42`。
- 下一安全动作：把 PG-384 的抽象选槽接入已审阅的本地 binder 做 dry-run/loopback typed replay；在第二独立实现、fresh role reset、candidate/reference/negative/replay 和 evidence SHA 全部满足前，不把具体 wire 记入训练、长期备忘录或 payload catalog，也不宣称通用漏洞能力。

### 2026-08-09：PG-384 模型选槽→本地 binder→typed loopback 展示

- 新增 `scripts/run_pg384_model_selected_binder_replay.py` SHA=`1cece52f233907ad354d1e32c3cac6b3418efa265f47f94322b093e143ab449e` 与测试 `tests/test_pg384_model_selected_binder_replay.py` SHA=`5c4aa4924f0a7c1a3a1cc29ac5e1398ba5a4045396b5c9893eb05510dc4b4ac9`。默认 dry-run；live 只接受 `PG384_LOCAL_EVAL=1`，仅启动本机 PG-348 loopback synthetic runtime，外网/ Docker 均为 false。
- dry-run 报告=`research/pg384_model_selected_binder_dry_run_v1.json` SHA=`db6527603e85c2d442123fbac675d33fc26023a32f07598ed6c532391a72e6c7`：48 个 holdout 行全部 abstract decode exact，6 行模型选中 safe Rule-IR、42 行 ASK、unsafe allow=`0`、binder reject=`0`。
- loopback report=`research/pg384_model_selected_binder_replay_v1.json` SHA=`8c3e5f5d7c3e7e375047a3e3ec1f404058cc987964666801a1c73fb62d8b0446`：48 行中 6 行进入模型选槽→reviewed placeholder binder→candidate/reference/negative/replay，confirmed typed=`6/6`、negative clean=`6/6`、replay consistent=`6/6`、42 行保持安全 ASK、unsafe allow=`0`。这是 synthetic evaluator 的闭环展示，不是 XSS/SQL/WAF 通用漏洞或可迁移攻击字符串。
- 报告 raw/context firewall：`model_context_raw=false`、request URL/body/response body/wire/canary 均未存储；具体 placeholder 只在 evaluator 内存展开。所有 promotion/training/memory/payload_catalog/vulnerability flags 仍 false；`second_independent_implementation=false`。
- 相关回归：PG-380/381/382/384 + binder=`24 passed`（包含 2 个既有 Transformer warning）。本次规则更新后 `research/improvement_rules.json` SHA=`e8bf63115eb8688ef9971a478b220363b24caafdc7175b2282ab91d8bde2ea85`；首阶段 next-token 熵硬门、后续层熵诊断范围保持不变。

### 2026-08-09：PG-379 双实现动态页面 smoke（严格阻断）

- 在用户授权的本地 Docker evaluator 中做了最小一 seed、两 route（GET/POST）×两套动态实现的 fresh smoke；固定 `network=none`、无外网、无持久化、角色容器逐项清理。报告=`artifacts/pg379/pg379_live_seed37901_get_post_smoke_report.json` 文件 SHA=`2c9a07efcae4cb4e33001b46d38220675cc638448f06fad782fd1234f6f1be50`（内部 report SHA=`51bd3a88891ba8d639a1a9285dd01773963379d4c9a50e8584d949c1eb329acc`），sidecar=`artifacts/pg379/pg379_live_seed37901_get_post_smoke_sidecars.json` 文件 SHA=`4d31fc205583857c0c3ed65633c4c8fc0f308a6a661ec7496da7ec9a84c2c2cb`。
- 16 个 role episode 均执行，但只有 `3/12` source rows 严格有效，`capture_failure=12`、`typed_role=0`，状态=`completed_incomplete_source_rows`；negative violation=`0`、raw response 未持久化、external network=`false`。失败原因只保留抽象 `runtime_RuntimeError`，不把工程失败伪装成漏洞阴性或训练样本。
- 因 fresh/adapter/typed contract 未闭合，`training_eligible=0`、`rows_written=false`、training/memory/payload/vulnerability promotion 全部 false。下一动作是修复受审阅的动态实现适配/路由投影后重跑同一最小 smoke，再扩大 seed/route；不扩大到公网或任意 WAF/persistent-XSS/反链 payload。
- 本轮只新增诊断证据，未改变分层熵规则：首阶段 next-token 仍执行 `relative entropy drop <= 0.25` 硬门；后续压缩/adapter 层只做有限 logits、非空类别、holdout 泄漏、slot coverage、ASK/repair/negative 的诊断检查。当前 `research/improvement_rules.json` SHA=`e8bf63115eb8688ef9971a478b220363b24caafdc7175b2282ab91d8bde2ea85`。

### 2026-08-09：PG-385 过滤反馈→受控 canary 字符串与 A800 候选结果

- `scripts/run_pg385_filter_repair_demo.py` 在固定本地 loopback fixture 中已经能展示最后一跳字符串：先发送被过滤的单层编码 canary，模型/抽象 reasoner 只读取 `filtered/encoding_filter/raw_delimiter_blocked` 等反馈，再选择 `double_layer_order_sensitive`，由 evaluator 临时绑定第二层编码字符串并发送；candidate/reference/replay typed=`1/1/1`、negative violation=`0`、action changed=`1`。`--show-wire` 只在显式本地 demo 时打印，报告不保存 URL/body/wire。该字符串不是通用攻击 payload，也不触碰外网、凭据、持久化状态或业务写入。
- 独立抽象数据集=`research/pg385_filter_repair_adversarial_dataset_v1.json` SHA=`9441a726a9659386f8c3e8f5d161675a5f5412c17ce4322d8d410d2e11909835`，构建器=`scripts/build_pg385_filter_repair_dataset.py` SHA=`ba27dc3004a2ce26e66d56ee48f635f8295a83c180e498b4eb0571086c1731bb`，测试 SHA=`f84a85362ed79869ae46e7909ce3eb7a21e5fe7681a5f3784dbc30acb010dd55`。共160条抽象记录（train=80、implementation_holdout=80），包含编码/语法/形状/解析边界/缺观测五类，raw payload、URL、响应体和 evaluator answer 均不入模，training_eligible=0。
- CPU wiring v2=`research/pg385_filter_repair_sft_cpu_smoke_v2.json` SHA=`d5b5439dccfd9189dff3814b301fe879d64d6cf4d56b6bae8156f1aa8b1bffd2`：sequence exact=0、slot=.357692、ASK=.75、repair=1、positive=1、negative false allow=16；仅 wiring 诊断，不能当能力成绩。
- 周末远程 A800 候选已在 GPU0（`CUDA_VISIBLE_DEVICES=0`，GPU1–7 未触碰）完成三 seed、4 epochs、d_model=512/8 layers/8 experts/hidden=2048；报告=`research/pg385_filter_repair_sft_a800_candidate_v1.json` 文件 SHA=`f07264f70920d79b429f0e835ada655948bb89d79e74e3c1797e8683c399c6fb`，内部 `report_sha256=e3703f849afeff4ef60719f1e9ae3a9fb579bdca7f7e79400d3fd50eb5ac1b02`。候选 worst slot=.794231、ASK/repair/positive=1、negative false allow=0，但 first-stage entropy drop=.484083（>25%硬门）、sequence exact=0；状态=`blocked_entropy_candidate_only`，训练/记忆/payload catalog/vulnerability promotion 全关。神经 checkpoint 仅登记远端哈希：seed38001=`19dfb73eeb7fedd652a58f77b90ee42dc519fcf55705db42d28ebbc16d284ede`、38002=`84c0b214d6d93c25afd4706b734a59e9ca4c75f28e7fd65cc7a8489ec8ef9c66`、38003=`f6265ecd19acb1d53c8debba5cb9ca8002bd9b2edc3e0c183dee75b8fddec98f`。实际解码检查中，神经模型没有稳定选出 double-layer encoding repair；可靠的最后一跳字符串仍来自确定性抽象 reasoner + reviewed evaluator template。
- 当前有效 `research/improvement_rules.json` SHA=`7f22770101efa0236934f5688e9eec77dc90ef3e19c4e4c1e82691c088aa5a10`。明早可展示“抽象反馈→受控本地 canary 字符串→typed effect→失败修复/复放”，不能说模型已经会对任意网址生成通用 WAF 绕过、持久化 XSS、反链或任意原始攻击字符串；若要升级为模型直接输出原始字符串，必须另行设计授权、模板绑定、negative/fresh/typed/evidence 和泄漏审计，当前规则明确禁止。
- 演示重跑后 `research/pg385_filter_repair_demo_v1.json` 最新 SHA=`03e84db31bf5b026ee61540df1aaa5e37ceecbd6b59723d359cbf698d91ddba7`；seed38003 候选 checkpoint 已从远端完整回传到 `artifacts/pg385-filter-sft/pg370_seed_38003.pt`，本地 SHA=`f6265ecd19acb1d53c8debba5cb9ca8002bd9b2edc3e0c183dee75b8fddec98f`，与远端一致。更新该报告哈希后的当前 rules SHA=`1b450b9fb9877eb79a1edfb14c0cce0f670055cb1c5b5c34fafb585c450955c3`。

### 2026-08-10：PG-385 模型实际选变体→本地 canary 复放

- 新增模型侧抽象变体选择器=`scripts/run_pg385_variant_selector_candidate.py` SHA=`3691404ae7fbef8b9884434f070096cd34cecf344b75a347a6c7370ca903dad4`，tests=`tests/test_pg385_variant_selector_candidate.py` SHA=`170c5d3f7b6419c79ef327dfc0bacb227a03e0ca3912201fcb1edfcbf8d4584f`。它使用 decoder-only CausalMoE backbone，只读 train-context tokens，训练 `encoding_ref/probe_variant_ref/repair_action/next_action/question/safe_to_send` 抽象 heads；不读取或保存 raw payload、URL、响应体、wire 或 evaluator answer。
- 周末 A800 GPU0 候选=`research/pg385_variant_selector_a800_candidate_v1.json` SHA=`e16c9da21f60bb6dd4364666b4a97e038346c13ef36c215a94c9ea1829a3bd46`，3 seeds=`38501/38502/38503`、20 epochs、d_model=512/8 layers/8 experts/hidden=2048。最坏 holdout：`variant_exact=1.0`、`encoding_exact=1.0`、`ASK=1.0`、`positive=1.0`、`negative_false_allow=0`；GPU1–7 未触碰，GPU0 已释放。promotion/training/memory/payload/vulnerability 全关。
- 新增真实模型演示=`scripts/run_pg385_model_selected_filter_repair_demo.py` SHA=`8adc7cb13871215719927bec58c073583dd617d3052e601bf3b7c46f186deea4`，tests=`tests/test_pg385_model_selected_filter_repair_demo.py` SHA=`94710553ceb04a49777667ce4455ffd6a380ca24f4bf0f48a2dee2bd0a27e9f7`。报告=`research/pg385_model_selected_filter_repair_demo_v1.json` SHA=`28bee104025ccca596893f90c28b126b6c2544d4fe41d9fe686621671c2f79a2`：模型实际输出 `encoding_ref=double_layer_order_sensitive`、`probe_variant_ref=one_variable_repair`、`repair_action=encoding`、`next_action=repair`、`safe_to_send=1`；随后 evaluator 将抽象变体转成 binder 的 source-attested alias，发送本地 canary，结果 baseline filtered=`1`、model variant selected=`1`、candidate/reference/replay typed=`1/1/1`、negative violation=`0`。具体字符串只在 `--show-wire` 的进程内短暂展示，报告不保存。
- 因此当前目标已达到“模型判断过滤反馈并选择抽象绕过变体，最终由受审阅本地 evaluator 生成并发送测试字符串”的可展示闭环；仍不能把它升级为任意网址、通用 WAF 绕过、持久化 XSS、反链或任意原始攻击字符串生成。当前有效 rules SHA=`7c07ac6b8537506d636f8c1ec8f51b2bc66118f0c9dd350a3c826c6a91a522bf`，AGENTS 本文件 SHA 在本次追加后需重新计算并作为下一次恢复校验值。

### 2026-08-10：PG-385 跨实现证据与前端功能展示

- 新增 `scripts/run_pg385_model_selected_cross_impl_replay.py`（SHA=`4383084ef51612a476822c768929b687772a726aed2fad07440e43665dde0fb1`）和 Node 独立 loopback fixture `fixtures/pg385/impl_b/server.js`（SHA=`ede5e6e7724732c5393d73b0c5e3032d42e9907345dfcd59cb8e55ad7689b181`）。固定 Python A/Node B 两实现各跑 GET+POST 的 candidate/reference/negative/replay，报告=`research/pg385_model_selected_cross_impl_replay_v1.json`（文件 SHA=`bfdf31b9211c8490f1336bb14610dcd6be80a34053bf6450f67d7ad681c57432`），4/4 model-selected、candidate/reference/replay typed=4/4、negative violation=0。该轮是 loopback process candidate-only：`docker_started=false`、无 image digest/容器 attestation，不能冒充完整科学 second-independent-implementation gate；promotion/training/memory/payload/vulnerability 全关。
- 前端新增 `/pg385`（`frontend/app/pg385/page.tsx` SHA=`34785aa6b4e3a2d906944156b058552c7106021918328ec4196f5649137f76ed`；`frontend/components/pg385-demo.tsx` SHA=`fff2978287898b829b6df61c621beb27d2baf9b6d3bc8e9af4723333c0bb58f5`；样式 SHA=`f4dce3dbb558217c485c0ff58b3b139aba8beb96310368d532a3a6a4c848fba8`）。页面展示模型选槽、脱敏过滤反馈、Rule-IR、GET/POST、Python A/Node B、candidate/reference/negative/replay、信息轴/熵门与 claim boundary；临时 wire 仅以 `<ephemeral-port>` 本地 canary 预览，不持久化原始字符串。主研究台操作区已增加“过滤反馈实验”入口（`frontend/components/research-dashboard.tsx` SHA=`caede70ea54a1e7a510ccc39dc18a9976fe1d2459bc66de96b02c4e8e59b4ba4`）。`frontend` 执行 `npm run build` 通过，静态路由包含 `/pg385`。
- `research/improvement_rules.json` 已登记 `cross_implementation_loopback_demo` 与展示入口，当前规则文件 SHA=`B4312C6A40F77EDB6CB8753032324903939BF249ABE5BFA13C1CF17D07DCDDC1`；展示语义仍限于“抽象条件→受审阅本地模板→typed effect/repair”，不能宣称任意网址、通用 WAF 绕过、持久化 XSS、反链或可迁移原始 payload。
- 规则随后补充 `long_term_memory_memo`：跨上下文长期备忘录的 canonical path 为 `AGENTS.md`，只允许 append-only 记录证据/失败/下一动作；训练、payload catalog 或能力晋级不等同于记忆写入。规则文件最新 SHA=`5E60D8613DD46493F1061FFF155BAC3FAD14421CB06F2AEAB84159FD99A436CA`。
- 本轮验证：`tests/test_pg385_model_selected_cross_impl_replay.py`、模型选变体/单实现 demo 共 `8 passed`；规则契约回归 `7 passed`；`frontend` 的 `npm run build` 通过并生成 `/pg385` 静态路由。当前没有启动 Docker、GPU 或外网服务。

### 2026-08-10：PG-386 fixture-bound 模型字符串生成闭环

- 新增抽象输出数据集构建器=`scripts/build_pg386_fixture_payload_generation_dataset.py` SHA=`cdd73ced277613ea5b29551bfaf9131ef29f8b3fca59d90d13ccd8758625d992`，数据=`research/pg386_fixture_payload_generation_dataset_v1.json` SHA=`4c468768da15eecc06cec35526136d44c769c29aefa797691717956779094700`。160 行（train/holdout 各80、GET/POST 各半），持久数据只含 `fixture_double_layer_value`/`ask` 抽象输出类和上下文；raw string、URL、wire、响应正文、evaluator answer 均不入模/不落盘，training_eligible=0。
- 新增 token-model 输出头与回放=`scripts/run_pg386_model_generated_fixture_payload.py` SHA=`1f300dbf3e0f9232520eef19affc13ab6e8bb6ee1b1d1fa4861a523cc32caa8f`，tests SHA=`fa117e8412617dd0a5b3b3a88f4286b042c039ca63a6111fc3e6117cf8dccfe2`。它复用 PG-385 decoder-only token backbone，增加 role-conditioned grammar head；模型实际选择并生成 fixture-bound value，adapter 只接受 `PG386_(CAND|REF|NEG|REPLAY)_0002%25253A` 形状，越界即阻断。candidate/reference/replay 每项 `4/4`，GET/POST×Python A/Node B，typed `4/4`，negative violation=`0`，replay=`4/4`；报告=`research/pg386_model_generated_fixture_payload_v1.json` SHA=`5c4aba9e91a18ebcc85bd00f2dfa3c122b45b5b965d0acd1a0b408c4070a9bed`，decoder checkpoint SHA=`ea57370656898b03b3988ddbe2fa94c69f126baf7dacd3e7451612e977d13827`。具体字符串只在 `--show-wire` 的 loopback 进程内短暂输出，报告只存长度/哈希/投影。
- 这是“模型判断过滤反馈→生成受 grammar 约束的本地测试字符串→发送→typed effect→负对照/复放”的能力证据，不是任意网址 payload、通用 WAF 绕过、XSS/SQL 执行、持久化、外连或任意原始字符串生成。Docker/GPU/network 全部 false；规则已登记 `fixture_bound_payload_decoder`，最新 `research/improvement_rules.json` SHA=`9473EDC94A59396D8442EA68E281B25AC23AC0976F3F45E172EFAE19AA32FEC1`。
- 前端 `/pg385` 已补上“FILTER / JUDGMENT TRACE”剖面：展示真实 fixture 的 raw delimiter/single encoding/double encoding/negative 四路决策表、bounded decode 算法伪代码、脱敏反馈 token、模型 Rule‑IR 选择和 grammar gate；页面源码最新 `frontend/components/pg385-demo.tsx` SHA=`8bc9898b7ba753864fea1c7504083c43f732af54a1f813271ee33ad47ab8657c`，样式 SHA=`8102ffc207d4479ffdd79c63a470a349d821cfb0a5b0870737fcbdf0cf1c5459`。本机 Next 服务已在 `http://localhost:3000`，GET `/pg385` 返回 200；该页面仍只调用静态证据/演示状态机，不把任意字符串发送到外部目标。

### 2026-08-10：PG-387 CTF-like 前端上下文投影与数据集

- 为了让模型学习“先读页面 JS 语境、再判断是否能做受控变体”，新增 `app/pg387_ctf_frontend_projection.py`（SHA=`5ee1b685611efcbbc5ccfbff10c5504cf4a6c29cd1a21564892b789ec29692cd`）。它对 bounded local fixture source 做一次只读投影，输出 `js_sink/loader_policy/state_policy/normalization/transport/response_shape/failure_shape/oracle_shape`，只保留 `source_sha256`，不把源码、URL、wire、响应体或 evaluator answer 返回给模型；外连、dynamic loader、动态代码、持久化 API 默认 `ASK/abstain`。
- 新增 `scripts/build_pg387_ctf_frontend_context_dataset.py`（SHA=`8f199db088cafe0367091bf11393b6e336596183d05e5f65e148a6defd216d99`）及抽象候选数据 `research/pg387_ctf_frontend_context_dataset_v1.json`（SHA=`c2b5f657e9a7152b824e510734e693f9ab5bb2348f38ef170f0fb99b7bf5ccfb`）：16 个 CTF-like sink/loader/parser/state 语境 × 2 implementation × 4 seed × candidate/reference/negative/replay = 512 行，GET/POST 各 256；split 保留 implementation_holdout。所有行 `typed_evaluator_observed=false`、`fresh_reset=false`、`training_eligible=false`、promotion 全关，不能冒充 live source rows 或漏洞能力。
- 新增测试 `tests/test_pg387_ctf_frontend_projection.py`、`tests/test_pg387_ctf_frontend_dataset.py`，专项 `6 passed`；前端 `/pg385` 增加 `01C · CTF FRONTEND CONTEXTS` 选择器，展示 JS context projection、抽象 token、`controlled probe / one-variable repair / ASK / abstain` 判定，`frontend/components/pg385-demo.tsx` SHA=`dabe2bafdbea36a2873ce9c365facae0bb5ab6accb17ccf7a7c695675566f947`，样式 SHA=`809a55560561ff87ff2e62ab10567321a40bd61dee0fe506fd85760358105a59`；`npm run build` 通过，GET `/pg385`=200，页面仍是静态演示，不启动目标或外网。
- 下一安全动作：若要把 CTF-like 模板变成训练/能力证据，先为每个语境接入 reviewed local dynamic implementation、fresh role reset、GET/POST、typed candidate/reference/negative/replay 和 evidence SHA；在此之前只可做 abstract candidate / ASK / evaluator-side demo，不扩展为任意 WAF 绕过、持久化 XSS、反链或任意原始字符串生成。
- 本次登记后的规则校验：`research/improvement_rules.json` SHA=`4fd0c7af9383640c653d637e5228ffc3b5a8e6d7bcfa1566e1ad4e513a530806`；`AGENTS.md` 的当前 SHA 由恢复流程外部计算，避免在文件内写自引用哈希。

### 2026-08-10：PG-387 process-only CTF context replay

- 新增 `scripts/run_pg387_ctf_context_process_replay.py`（SHA=`5880bda3c22531c10dd78ca2fba3009091370a00a6c485a91ca4f84e2b8edc10`）与 `tests/test_pg387_ctf_context_process_replay.py`（SHA=`a5aac7ce9fa086c81403f569c0dcf9034cb3603c516e3072df03e947cf4fce51`）。它只对四个本地 CTF-like JS 语境做 bounded process-only replay：double-decode text sink 进入 loopback canary，script loader、persistent state、dynamic code 三类保持 ASK/abstain；candidate/reference/replay typed=`3`、negative violation=`0`、action_changed=`4`、fresh reset=`4`、ASK=`12`。
- 报告=`research/pg387_ctf_context_process_replay_v1.json` SHA=`90bc64ef17aa4667ad683eacc9a728f4c6483ff62d24187467ab9d5d4a99a8c2`。报告只含抽象 tokens、bounded typed projection、reset/evidence hash；具体 marker/wire 只在 `--show-wire` 进程内短暂存在。`docker_started/network_contacted/gpu_touched/training_started=false`、`training_eligible=0`、promotion 全关；`3 passed`。
- 这一步证明“JS 语境读取→ASK/repair→本地受控发送→typed projection→fresh reset”工程链可运行，但决策来源仍是 `abstract_context_policy_not_neural_checkpoint`，不能说 PG-387 模型已经学会；下一步必须把同一语境接入真实 token-model checkpoint、第二独立动态实现和 image-attested fresh Docker replay，再申请 A800。
- PG-387 process replay 登记后的前一版 rules SHA=`8ad39c8d8b1d88f44e37e875614628f6dcf17affdb442f04a6b6abfdc5e02dd5`；`AGENTS.md` 的当前 SHA 由恢复流程外部计算，避免在文件内写自引用哈希。

### 2026-08-10：PG-387 token model decision candidate

- 新增 `scripts/run_pg387_ctf_context_token_candidate.py`（SHA=`61f5b8febebc8f943849e29399080584a9e0ff0278c961f3338efebf9db7ec3e`）及 `tests/test_pg387_ctf_context_token_candidate.py`（SHA=`47659884c0120202baaf4961cebe44f78a235820b24434b4c550bfa7b9f7246a`）。它复用 decoder-only CausalMoE，只读 PG-387 abstract JS-context tokens，head 输出 `next_action/repair_action/probe_variant_ref/ask_reason/safe_to_send`；train-only vocabulary 与 holdout gap 在 optimizer 前检查。
- CPU smoke 报告=`research/pg387_ctf_context_token_cpu_smoke_v2.json` SHA=`262a53ce0e04574b8d0e56252576b7cc9e7e3fb94b94ff155920a0e83aa21ca2`：3 seeds，train/holdout 各 128，8 epochs，d_model=64/2 layers/2 experts；worst next_action/repair/probe/ASK/safe 均 `1.0`、negative false allow=`0`。这是同语境合成留出上的 wiring evidence，不是独立实现泛化或漏洞能力；optimizer 只在 CPU，GPU/Docker/network=false，`training_eligible=0`、capability/promotion 全关。
- 相关 PG-387 + rules 回归 `15 passed`。当前只允许把该结果作为 token model 的上下文决策候选，不能据此提交 A800；A800 前置仍是真实第二动态实现、image/source attestation、fresh role reset、GET/POST candidate/reference/negative/replay、typed evidence 和 cross-implementation holdout。

### 2026-08-10：PG-387 JS 语义投影与 PG-379 动态整页 smoke

- `app/pg387_ctf_frontend_projection.py` 已扩展为 append-only JS 语义 overlay：`js_source`、`js_parser`、有序 `js_normalization_step`、`js_filter_shape`、`js_guard_shape`、`js_control_flow`、`js_event_shape`、`js_ast_shape`、`js_source_to_sink`、`js_sink_context` 和持久化/动态代码标记。它仍只保留 source SHA，绝不返回源码、URL、正则、wire、响应正文或 evaluator 答案；不安全 loader、持久化状态和动态代码继续 ASK/abstain。模块当前 SHA=`43c47dbbd4013a15faff1b25fa11c4f4da9cab1ed3175955cb65aaa0faf3424`，projection tests=`6 passed`。
- `scripts/build_pg387_ctf_frontend_context_dataset.py` 已把同一语义 overlay 写入 512 行抽象 CTF candidate（16 case×2 implementation×4 seed×4 role），context token 与 metadata 对齐；dataset 当前 SHA=`722cea13a94357e3edc96f1d08122ffcbf36fa6d3bb1b3a9514268ffd02b887`，dataset/projection/rules focused=`12 passed`。状态仍 `abstract_ctf_candidate_only`、training/promotion 全关。
- PG-379 双实现动态 smoke（单 seed、2 route、16 role episodes）完成：runtime=`16/16`、source rows=`12/12`、adapter-valid=`12/12`、typed=`12`、negative violation=`0`、failure/action-change=`4/4`、capture failure=`0`；image attestation、fresh reset、network-none/loopback、13-slot target 和 context firewall 均通过，但 `training_eligible=0`。该 smoke 的两条 route 实际 JS 脚本计数为 zero，因此只能证明“JS 轴被结构化观测/可标记为空”，不能证明模型已读懂真实业务 JS。
- 当前解释边界：黑盒可从 DOM/响应/失败差分推断，但遇到解码顺序、sink 类型、过滤调用链或持久化状态时，必须给模型 JS 语义投影；这不是把原始 JS 喂给模型。源码 tokenization 仍是语义/形状层，非逐字符或完整 AST 生成。
- 使用更新后的 512 行语义 JS token 数据做 CPU-only token-model smoke：`research/pg387_ctf_context_token_cpu_smoke_v3.json`（SHA=`196fba54c22efd7300ecfbe4e461ff13cc8fc5d851f3107d42cc8b5983fe4fa7`），3 seeds、train/holdout 各 128、train-only vocab=92，next_action/repair/probe/ASK/safe 均为 `1.0`、negative false allow=`0`；这是同语境 wiring evidence，GPU/Docker/network=`false`，training/promotion 全关，不是独立实现能力证明。
- 当前 rules 校验（含 token model candidate 登记）：`research/improvement_rules.json` SHA=`309ae0581f9ac5272daabff90fc63e922621d24c78afc8818469f26cbb521493`；`AGENTS.md` 当前 SHA 由恢复流程外部计算。

### 2026-08-10：PG-388 逻辑/业务漏洞不变量实验与动态本地靶场

- 新增 `app/pg388_logic_invariant_projection.py`（SHA=`09d5d5fe852bb9e27f9ac351f83a575555cd4cfab535211325db1de83b1c8aeb`），把安装、交易、优惠券、账户规范化、找回绑定、2FA、验证码、Session、水平/垂直越权、可预测 ID、执行顺序和敏感字段投影统一成不变量/前置条件/状态转移/反事实/失败反馈/修复动作。缺观测为 `ASK`，negative 为 `abstain`，模型上下文不含原始值或 evaluator 答案。
- 新增抽象数据集 builder=`scripts/build_pg388_logic_invariant_dataset.py`（SHA=`3a4d1878e47dcc2006f55b30fc3b9e5219a7e07375c92df0dd35a5eec5232a5f`），产物=`research/pg388_logic_invariant_dataset_v1.json`（SHA=`eb8b39e20462f0d6a0435d365b2591f8a2c3be6f5ef858c420d676f66d663127`）：56 case×2 implementation×4 seed×5 feedback×4 role=`8960` rows，train/implementation holdout 各 4480。新增只读 audit=`scripts/audit_pg388_logic_invariant_dataset.py`，audit status=`passed_candidate_audit`，但 `training_eligible=0`、typed evidence/live rows=0、promotion 全关。
- 新增纯 in-process process replay=`scripts/run_pg388_logic_invariant_process_replay.py`（SHA=`4b45c71eaaf3852211502d47e696f8c5d10846a9b9bdba0dd8847e1f2079806d`），报告=`research/pg388_logic_invariant_process_replay_v1.json`（SHA=`66f279759476d3184ad3ee3c193473141d2ce4eccd40b3a7cb57f95853b2ce57`）：672 episodes，typed effect 504，negative 168 且误放 0，failure/action-change 336/336，fresh reset 672；只证明抽象状态机合同，不是 live 漏洞能力。
- 新增动态后端 fixture=`fixtures/pg388/logic_lab.py`（SHA=`d54c02bb7bcfcfcabb4725a7032862f0130302980a4e5b74fd990af94b9250ae`）与 Dockerfile/README。它只接受 case/role/feedback 枚举，运行时内存状态可重置，默认 network-none/loopback-only、无持久化业务写入、无凭据/任意值/外连；真实容器启动仍需审阅 immutable base digest 和显式操作员开关。前端通过 `/pg388-api/health` 探测后端，未启动后端时自动回退为静态展示。
- 新增一体化展示部署：`frontend/Dockerfile` SHA=`28ec959fa4cd43ad985a1641bc624d18fc8a12a43313114b274b945bee503331`、`frontend/next.config.mjs` SHA=`2c8de0a2053938c57603efaf5e4604094d0633312bdadce62efc321178ead8db`、`docker-compose.pg388.yml` SHA=`5292c6595b1f29b1e25fd1043bb8e3513481d5367a789869d0a9bb41f01d815e`。compose 只把前端发布到 `127.0.0.1:3000`，后端在 Docker `internal: true` 网络内；这条是展示部署，不能替代 evaluator 的 network-none 合同。
- 新增 token-model candidate=`scripts/run_pg388_logic_token_candidate.py`（SHA=`f0f77236bfb84362762cef66b8a00baaac6ac5bf771ba729501c22799ffa501c`），只读 PG-388 abstract context，输出 11 个逻辑 Rule-IR 头。CPU smoke 报告=`research/pg388_logic_token_cpu_smoke_v1.json`（SHA=`542c3ec0ea0b8828f8a460f8d3d2ec6e31f165ba30bedc343e6149de6175cc22`）：3 seeds、train/holdout 各128、next_action 最坏 0.75、ASK=1.0、negative false allow=0；仅 wiring evidence，GPU/Docker/network/wire 全 false，promotion 全关。
- 新增前端 `/pg388`：`frontend/app/pg388/page.tsx`、`frontend/components/pg388-logic-lab.tsx`、样式，展示 14 个核心案例和后端 56 个不变量合同的入口，包含 invariant、precondition、counterfactual、ASK/REPAIR/ABSTAIN、candidate/reference/negative/replay、typed oracle 与 fresh reset；`npm run build` 通过，页面只展示抽象状态和本地 canary 合同。组件当前 SHA=`88fb0b984998de2a4e19f02f06c8699a339302fab94ca4e30240b18b16a2aa4a`。
- PG-388 focused regression=`17 passed`（含部署合同）；规则已登记 `pg388_logic_invariant_lab`，当前 rules SHA=`f81854aa4b45ae9c23689d91267af8424549ccd3118bdbe0313d88bf52ca199e`。后续安全动作是接入第二个 reviewed dynamic implementation、fresh role reset、typed candidate/reference/negative/replay 与字段/熵审计；不把该候选数据集或静态演示冒充通用逻辑漏洞能力，也不开放任意目标、外连或可迁移原始攻击字符串。

### 2026-08-10：PG-388 前后端展示部署联动

- `frontend/next.config.mjs` 新增 `/pg388-api/*` 本地代理，`pg388-logic-lab.tsx` 通过 `/pg388-api/health` 探测后端；后端不可用时明确显示 `offline / static fallback`，不伪装成 live 结果。
- `docker-compose.pg388.yml` 提供展示部署：前端仅发布 `127.0.0.1:3000`，后端与前端在 Docker `internal: true` 网络中，Python/Node base image digest 必须由操作员显式提供；compose 配置静态校验通过，未启动 Docker。
- 这条 compose 仅用于本地展示联动。逻辑 evaluator 仍以 `network=none`、fresh disposable、无持久化和 abstract Rule-IR 为硬门；未进行 Docker/GPU/外网运行。
- 后端镜像已用本机 `python:3.11-slim@sha256:90744cff…` 在 `--network=none --pull=false` 下构建并完成容器内 manifest smoke；前端宿主 `npm run build` 通过。随后前端镜像也用锁定 Node digest 构建成功（image manifest=`sha256:442196850b9d8234e5084993714f38945973841ae0d43336e42ddf7e225192ea`）；npm 构建过程仅访问包 registry，不接触任何靶场目标。
- 容器 smoke 证据=`research/pg388_logic_backend_container_smoke_v1.json` SHA=`30c124ad45fd96c4e22be7617189fe65042140db146ac52ec546d0e03a9b2cd4`，记录 image ID、network-none/read-only/tmpfs、case_count=56 和全部 promotion/training=false。

### 2026-08-10：PG-388 展示联动最终复核

- Next rewrite 对 Docker standalone 的运行时环境不可靠，已改为 `frontend/app/pg388-api/[...path]/route.ts`（SHA=`352974acdd1e40194583cd4c85626ad8dc8abcd2622aab4858851fa6fd634a3d`）运行时代理。它只允许 `health`、manifest/cases/reset/observe/episode 五类抽象路径，只接受 `127.0.0.1/localhost/::1/pg388-backend`，请求≤4 KiB、响应≤64 KiB；非法路径、外部主机和超限均 fail-closed，不记录原始值。
- compose 修正为后端只连接 `internal: true` 的 `pg388_internal`，前端连接该内网和独立的 `pg388_display` 展示网络；只有前端绑定 `127.0.0.1:3000`，后端没有 published port。compose SHA=`6954d3cb6495884c413046e7be0e10cbd09f096d85e1e21c59838a9242bb206e`，Next config SHA=`5913fa5ef4c1f65fac55f954d1bc81e7fd33018ecc77917f3116faf8357d91b7`。
- 已停止占用 3000 端口的旧本机 Next 开发进程并实际重建/重启 compose。端到端结果：`GET /pg388`=200、`GET /pg388-api/health`=200，返回后端抽象 health（external_network=false、persistent_storage=false）；容器内后端直连=200。展示联动 smoke 不等同 evaluator target contact，未启动训练/GPU/外网。
- 部署合同测试 SHA=`62327e76c5882d4b9362ddcc675bdFA52ea454661df963321be7f066d122d3d4`（实际文件 hash 以外部校验为准）；展示 smoke 只证明页面到抽象后端 health 的联通，不是 evaluator target contact。规则文件更新后的 SHA 以本轮外部 `Get-FileHash` 为准，AGENTS 不自引用自身 hash。

### 2026-08-10：PG-388 展示镜像最终锁定

- 由于运行时代理和 compose 网络修正后重新构建，当前前端镜像 manifest=`sha256:cb46a7f86d01c47f79e4095a4723aacdebe0169ef5ab58e49699f3e6a788ebe8`；后端镜像保持=`sha256:b3f7d7577f35897403582a016abe22b7ee508788e29d9d995ebbd1425b67eeb1`。规则文件当前外部 SHA=`d4946eb256e2346707d8079e48d806b1aed67b2bcbf0eb9f42d9250cf7802d29`。

### 2026-08-10：PG-388 前端真实抽象回放接线

- 前端组件已从纯时间线动画改为真实调用受限 `/pg388-api`：`POST /api/reset` → `POST /api/observe`（missing/ASK）→ candidate/reference typed episode → negative typed episode → replay typed episode；只提交 `case_ref/role/feedback_state` 枚举，不提交业务值、URL、凭据或原始响应。组件 SHA=`3d4375f81dc6f723a7f83d84533dc86ab08a2c9dd5f4a2224554c05c779e39ec`，Next 构建通过。
- 页面运行态烟测：`/pg388`=200、`/pg388-api/health`=200、`/pg388-api/api/cases`=200 且 56 cases；一条 `purchase_price_binding` 交互链返回 `fresh_reset → ask → candidate_typed/reference_typed → negative_clean → replay_fresh_required`，`safe_to_send=false`、`state_delta=zero`。该链仍是抽象动态 fixture，不是漏洞利用或任意目标测试。
- 重新构建的前端 image manifest=`sha256:4bee2f7d51c37996b457854b76f65023e056d13f9bd452fed8aa11641061b389`；当前 rules 外部 SHA=`83f90e86b9e1430e3e9648fcc56f89c275387fdf2f289876340e34f141a4c86e`。compose 保持运行供展示，后端没有 published port。

### 2026-08-10：PG-388 交互烟测证据固化

- 展示回放证据=`research/pg388_frontend_interactive_smoke_v1.json` SHA=`e437964f2b882cb1c00086601c0bbaba7867953b4d1bb61a288b7a22313e7384`；只含抽象 reset/ASK/typed/negative/replay 状态，不含原始请求或响应。
- 规则文件最新外部 SHA=`f8e3e47f011368044ce79e74e93711cda1c6ffa85befb0ec1ff1ea76270231a8`。当前 compose 仍为展示运行态，下一步若扩展研究应先增加第二独立实现和跨实现 live evidence，而不是把这次抽象烟测直接当作训练金标准。

### 2026-08-10：PG-388 失败反馈→修复展示更新

- 组件最终 SHA=`cd0631a9a3b7d58341f06ac9e22b389f860a4e439b5b1acf35c97fcd3ec90838`，交互链新增真实 `invariant_mismatch → repair_action`，随后才执行 candidate/reference、negative 和 replay。
- 交互烟测报告已更新为 SHA=`ad88d8e35cbaeaf42798f1164d95979ea69998cedb67c7ffefb61c903c852428`；前端最终 image manifest=`sha256:71079576ec2c9133d872720523a610304b741cbbfb85744e584b5bafd06bc0cb`。规则文件当前外部 SHA=`658a449b8cf24947c07906b6fd148c232bdd3d40e0633f8e5ab1b0ca4882ff7a`。

### 2026-08-10：PG-388 4.13 分类覆盖审计

- 新增只读审计 `scripts/audit_pg388_logic_taxonomy.py`（SHA=`d52fd71d57318aa3acb574da0ad5dacd77ee088e08fda3bd7a728bbeb7bec06c`）和测试（SHA=`4fdc1b87d9e05382a85495dd5ce3506192857c9b124fd7223de0e208875782fe`）。
- 报告=`research/pg388_logic_taxonomy_audit_v1.json`（SHA=`501d231de9ea5505021168b81ad3ad5c2cc5d9285f8342e1501a4a186b4196e6`）：56/56 个主分类锚点已覆盖；明确列出 10 个待补诊断项（OAuth/激活链接/CSRF 2FA、CAPTCHA 预测性/响应泄露/客户端校验/投递滥用、Session 猜测/伪造/泄露）。这不是训练许可，`training_eligible=0`、promotion 全关。
- 规则文件当前外部 SHA=`9391bb018a2a302b56ecdc1c8f4bc14e1c0e89962358ca393f07d6e742900532`；后续新增案例必须使用 reviewed local implementation + fresh role reset + candidate/reference/negative/replay typed evidence，不静默回填旧数据。

### 2026-08-10：PG-388 4.13 诊断缺口补充（candidate-only）

- 在不改动冻结的 56 类 v1 数据集的前提下，`app/pg388_logic_invariant_projection.py` 追加 10 个抽象合同：`oauth_second_factor`、`activation_link_second_factor`、`csrf_disable_second_factor`、`captcha_predictability`、`captcha_response_exposure`、`captcha_client_validation`、`captcha_delivery_abuse`、`session_guessing`、`session_forgery`、`session_leakage`。模块 SHA=`1268581d212f0301b0602024c85d082c1c29add304416da4d3829f78129d7ee9`；原 `LOGIC_CASES=56` 保持不变，新增项通过 `SUPPLEMENTAL_LOGIC_CASES/ALL_LOGIC_CASES` 单独访问。
- 补充 builder=`scripts/build_pg388_logic_supplement_dataset.py` SHA=`d2254d40210f3148869fcc48a1b40e48e5b0c224b43443af55b1d45c64bae2fd`，数据=`research/pg388_logic_supplement_dataset_v1.json` SHA=`df0a1a3fcf7d1f6cdb0876e11a8c257900903ba35b0c0eb0302bb55cc668d366`：10×2 implementation×4 seed×5 feedback×4 role=`1600` 行，train/implementation_holdout 各800，training_eligible=0。
- 补充只读 audit=`scripts/audit_pg388_logic_supplement_dataset.py` SHA=`fec5d46a8afec8e7146fbb5f80925b22b44108d178b817a8338abd902aac21a7`，报告=`research/pg388_logic_supplement_dataset_audit_v1.json` SHA=`17d90519b497a421b95e62b148187a6e5b658c18c6013f731ce586d02bcdfa86`，状态=`passed_candidate_audit`，invalid/raw hits 均0。
- 补充 in-process replay=`scripts/run_pg388_logic_supplement_replay.py` SHA=`b7373ccaadcf811b03dd4de2def16882f15012b5324cae1ca68a180f20783905`，报告=`research/pg388_logic_supplement_replay_v1.json` SHA=`35984de40981f52d3a22c5fbfbd61a2fcb238e7a6f463c065110e176011af703`：120 episodes、typed=90、negative=30、negative violation=0、fresh reset=120；纯进程内，不接触 Docker/网络，promotion 全关。
- 动态展示后端新增只读 `/api/supplemental-cases`，继续接受抽象枚举，`safe_to_send=false`、无业务写入；后端 SHA=`5d7efd39075a95978a9d75414d0576ac20f97daebe1ebeef1ac87b03f4a3a951`，本地镜像=`sha256:376b436a8d02cef6b2683d81e57a7a5c64700e4e2ef637436f99d87d4f155ba0`。前端 proxy 仅新增该 allowlist 路径，SHA=`4993053b85c25233df10659e36e203f0123d0a727f9985afbebf6a1d3b2bd5e9`；组件 SHA=`aa9d591bb60cf74c394660b44aebb45fae3954db1c9df4f1f231af6b0bcaa8ac`，镜像=`sha256:77174a931bde8b4563b3d05c0578913a2ee4fa3291d7a4cacb37597652e0d21f`。
- 已重建/重启 `pg388demo` 展示 compose 并验证：`/pg388=200`、health=200、原 56 cases=200、supplemental endpoint=200 且返回10项；页面包含 taxonomy-gap 状态。交互烟测报告更新为=`research/pg388_frontend_interactive_smoke_v1.json` SHA=`e62d5e42a035c60fa56e525e04ad9d0a173db5e2e8b562ce3b51f5efbb554ce3`，raw context/wire、外部网络、训练和晋级仍为 false。
- 相关回归：补充测试=`tests/test_pg388_logic_supplement.py` SHA=`c15da414efd3df264abfcadf4f651d45f3bf54b71e70ac0fc7e77efe457c68f7`；补充/fixture/deployment 组合=`9 passed`；前端 `npm run build` 通过。规则文件本轮最终外部 SHA=`54acd9d3f2ed598e5e65e3259c28a627ab915b5622fe66351a291929329b4fb8`。这些行仍是抽象候选/展示合同，不是 live 漏洞利用、通用 WAF 绕过或任意原始 payload 能力。

### 2026-08-10：PG-388 前端案例矩阵扩展

- `/pg388` 前端目录从14个静态展示案例扩展为24个可选案例：14个核心 + 10个 taxonomy-gap 补充项。补充项可直接触发同一枚举受限的 reset→ASK→failure repair→candidate/reference/negative→replay 链，不改变 raw/context firewall。
- 前端组件最新 SHA=`9d01d340ae2477f246823f1a6dba42c3cf00ce91b699601b9aa345a470e2dc4f`，重新 `npm run build` 通过；前端镜像=`sha256:6fce00f01495ae28edb6f6412d6ef55b725d9776b6c37cb7ec4f982c2cbef801`。
- compose 已重新创建并验证 `/pg388=200`；展示 smoke 报告=`research/pg388_frontend_interactive_smoke_v1.json` SHA=`7150c15a7c812019ca858b5e4672b8b6188344f816084063241441f320de6068`，目录计数=`24`，补充 endpoint=`10`。规则文件本轮外部 SHA=`b7f3d85878c410c766d28ec5b008635f3e305095cf2ac92805da4d1a040fdb01`。

### 2026-08-10：PG-388 靶场说明同步

- `fixtures/pg388/README.md` 已说明 56 个冻结核心合同、独立10项补充 endpoint、24个前端展示案例和 enum-only 输入边界；README SHA=`305a933dea9f33b7f9f9cb27a3de248e5a28c0259b68b2a46c05fee142ea635e`。
- 规则文件相应更新后的外部 SHA=`8bf02c1f11a9a3ac153fcbae778049976e0d2c9879137f7d5ab16c2baccd85f0`。展示容器继续是 disposable in-memory simulator；它不是任意网址、真实支付/账号/验证码或持久化业务系统。

### 2026-08-10：PG-388 补充合同 CPU wiring smoke

- 复用抽象逻辑 token candidate runner（仅放宽 dataset status 接受 supplement candidate）对补充数据做 128 train/128 holdout、3 seeds=`38801/38802/38803`、2 epochs 的 CPU smoke。报告=`research/pg388_logic_supplement_token_cpu_smoke_v1.json` SHA=`f52f9baaab6318e3b96a0bac0ef9cfa5a4dc1d68ea52770eab5188ba469f65de`；runner SHA=`8764ad95b1f79b2f7ab2550f3d185fd7e09d12928cdb3f1437b3c33d843dbc20`。
- 结果是 wiring evidence：train-only vocabulary gap=0、ASK recall=`1.0`、negative false allow=`0`，worst next_action=`0.679688`；optimizer 只在 CPU，GPU/Docker/network/wire 全 false，`training_eligible=0`、promotion 全关。该结果不证明 live 逻辑漏洞能力，也不等价于 A800 训练。
- 本次 rules 外部 SHA=`c046d66b78ffcc0baa063801117392baf035b0d98df95b21926e589e2f445d6d`；`tests/test_pg388_logic_supplement.py` 现在还校验该候选报告的 CPU-only 和 fail-closed 字段。

### 2026-08-10：PG-388 taxonomy audit 同步补充合同

- taxonomy audit 已改为读取 `ALL_LOGIC_CASES`，报告=`research/pg388_logic_taxonomy_audit_v1.json` SHA=`2a71b4de7f6b2e33f7b7523d50ec5b0afa215fcef958fd524f0d89291d8b5c45`：核心56 + 补充10 = 66，`missing_anchor_count=0`、`diagnostic_gap_count=0`、`candidate_only_count=10`、status=`passed_candidate_coverage_all_anchors`。补充合同只是覆盖清单，仍没有 live typed evidence 或训练资格。
- 审计器 SHA=`66f744b69aba4e48469d7962e9bea02b8c9149d45a1c2213a1dab4adb41ebac5`，测试 SHA=`13c6d86d94fd29c28d127a74255d0ff1c46e60630e92329e29bc281fe5ba47a3`；专项测试=`2 passed`。
- 同步后的 rules 外部 SHA=`a2109c55898cab9603c04c9e1e65cda317a0ddb44da762fb4c34ad94ee77c504`。

### 2026-08-10：根 README 展示入口

- 根 `README.md` 已加入 PG-388 Docker compose 启动和 `http://localhost:3000/pg388` 展示说明，明确 24 案例、抽象 enum-only 后端与 PG-385/386 fixture-bound 边界。README SHA=`ec3c37e603fb44719467e42807c63db50ebcfd51d73c4bba85995af5b89f8c4e`。

### 2026-08-10：GitHub 提交预检

- 当前工作目录不是 Git 仓库，没有 `.git` 和 remote；未执行 `git init`、commit 或 push，等待用户提供 GitHub repository URL、可见性和推送授权。
- `.gitignore` 已加入两个超过 GitHub 单文件限制的历史研究数据：`research/pg360_slotwise_dataset_v1.json`（约348MB）与 `research/pg361_syntax_slotwise_dataset_v1.json`（约682MB）；`artifacts/`、前端依赖和构建缓存原本已忽略。
- 推荐提交内容为源码、`fixtures/pg388`、`frontend`、`scripts`、`tests`、`AGENTS.md`、README 和小型审计报告；大权重/缓存只保留 SHA 与本地归档，不上传。

### 2026-08-10：GitHub 大文件与演示资产发布约定

- 当前工作区已是 Git 仓库，分支=`main`，`origin=https://github.com/wrench1997/blackboxanalyze.git`，最新本地提交=`ca9459e`；工作区干净。此前旧的“不是 Git 仓库/无 remote”预检记录仅保留作历史，不代表当前状态。
- 新增 `research/pg388_demo_asset_manifest_v1.json`，登记 PG-388 抽象数据、审计、可选候选 checkpoint 的路径/字节数/SHA-256；新增 `scripts/verify_demo_assets.ps1`，接收方下载后必须先做路径、大小和哈希校验。
- 新增 `docs/GITHUB_DEMO_RELEASE.md`：GitHub main 放源代码与小型 manifest；Release 放明天演示所需的少量资产；A800 只作训练/推理缓存，canonical copy 必须留在本地归档或受控对象存储。大权重不因位于 A800 就自动获得训练、记忆或晋级资格。
- 本轮未上传原始 payload、wire、响应正文、凭据或外部地址；GitHub 推送需在网络可用时显式执行 `git push origin main`，Release 资产另行发布并再次校验哈希。

### 2026-08-10：GitHub main 同步与可选大文件校验修正

- 用户已明确授权推送；`origin/main` 当前与本地提交 `ebaa94dbad4da1f65864712653f837c9dc04d0d5` 一致，工作区干净。该事实覆盖上方“等待授权/未执行 push”的历史预检记录。
- GitHub API 当前没有 `v0.1-demo` Release；不能在接收流程中假定该 Release 已存在。源码克隆可直接启动 `/pg388`，Release 下载命令仅在实际创建后使用。
- `scripts/verify_demo_assets.ps1` 新增 `-AllowMissingOptional`：源码克隆时校验 Git 内的小型资产，并把 `distribution=release_or_a800_cache` 且 `required_for_frontend_demo=false` 的缺失 checkpoint 明确列入 `missing_optional`；不允许把缺失权重当作已验证。所有资产齐备后必须去掉该开关再校验。

### 2026-08-10：PG-388 三类本地业务状态机 canary

- 为避免 PG-388 只有抽象投影，`fixtures/pg388/logic_lab.py` 新增 enum-only `/api/canary`：`nonce_replay`、`coupon_reuse_boundary`、`subject_resource_scope` 三类 disposable 状态机，输入仅 `case_ref/role/phase`，输出仅状态桶、差分桶和 typed local sidecar；implementation SHA=`e0b7e634e854b9e1dc2ae86da2e785afd42a427660e9abc981edb4370d5f0a23`。
- 每个序列执行 `baseline → candidate → reference → negative → replay`，在内存中观察重放/重复优惠/跨主体访问的差分；3 次 fresh reset、15 次 typed observation、4 次 candidate replay/scope effect、3 次 negative clean、unsafe allow=0。报告=`research/pg388_logic_canary_smoke_v1.json` SHA=`adaddd710b8250860096b458cb1dd580d7ff9018ca647fb9ac8eb40e5b716455`。
- 前端 proxy 已 allowlist `/pg388-api/api/canary`，组件会把 `local_canary=typed_state_violation/typed_clean/abstract_only` 显示在 live projection 中；API route SHA=`1dc81fa08fb0fc2fbe35c17f8e3ca521b2cb99d312e2024ff7f1cdd3672eec01`，组件 SHA=`9dc615fdb6594d938fd8dbff640e2f6838110ac6b755a5301e9278576527653f`。
- 该 `vulnerable_effect` 只表示 disposable simulator 内的 typed 状态差分，不是现实应用漏洞、通用 payload 或任意目标能力。`safe_to_send=false`、target/network/wire/persistent storage 全 false，训练/记忆/晋级全关闭。规则文件本轮 SHA=`964b511c881db6c7937f06494b44c26bfa6c7bd2ee9d80b9dee880438906b631`。

### 2026-08-10：PG-388 canary 轨迹候选与 CPU wiring smoke

- 新增 `scripts/build_pg388_logic_canary_trajectory_dataset.py`（SHA=`12f5ef77e5e1592a0b26e38268a1249c7d45bd9e9e9ae6d49622c0af5935e803`）和 `scripts/audit_pg388_logic_canary_trajectory_dataset.py`（SHA=`744c6aae08d440e5228cd19008b992f1813fe67f3f48f4a72c98f2e603758eee`）。数据=`research/pg388_logic_canary_trajectory_dataset_v1.json` SHA=`50b0e6933da17046de1302f9aa8da65b0b752a809dcef3703443c38156ad5192`：90 行，train/implementation_holdout=`45/45`，3 cases×3 seeds×5 phases，context 仅 state/invariant/phase/role，typed effect 只在 target/evaluator sidecar，audit=`passed_candidate_trajectory`。
- 复用 `scripts/run_pg388_logic_token_candidate.py` 做 45/45、1 epoch、d_model=32 的 CPU wiring smoke，报告=`research/pg388_logic_canary_token_cpu_smoke_v1.json` SHA=`c18790df76cfb377c3b255803ad666658124f6d6fd819803cd381ea76329719a`；train-only vocabulary gap=0，optimizer 仅 CPU，GPU/Docker/network/wire 全 false，`training_eligible=0`。结果中 next_action/repair/ASK 并不都高，保留为诊断而非包装成能力。
- 规则文件同步登记 trajectory dataset、audit 和 CPU smoke；manifest 已登记三个小型资产。该轨迹集不产生 raw payload、raw wire、真实业务值或通用漏洞标签，promotion/training/memory/payload/vulnerability 全关闭。

### 2026-08-10：PG-388 轨迹去重审计修订

- 轨迹 builder 已把 train implementation 与 holdout implementation 的历史顺序显式区分为同一 ontology token 的不同有序序列，避免跨 split 精确 context 复用；当前 builder SHA=`880bb4814a0513476abf8b6d89556862a34982c9193c0cedb5d7fe6e2b7d596f`，dataset SHA=`b600b849466d029319765953c43eafa0e15ee587bfa62f52a28aaf3d03198bbf`，90 行、train/holdout=`45/45`。
- audit 已新增 `cross_split_context_overlap` 与 `cross_split_context_target_overlap`，当前均为 `0`；audit 报告 SHA=`04cf1b52dfd9a700d5a414340214ebfa7236c8e392d45164c793e65a2dd9bffd`，audit 脚本 SHA=`ba3fd64e338e5bc652ea165b831b914430819f401983ccba829c74aca51754a5`，状态仍为 candidate-only，训练资格与所有 promotion 继续关闭。
- 1 epoch CPU wiring smoke 已按新 dataset 重跑，报告 SHA=`47a591235e24e15f6ab3dbe8b4d2d7d60bb7bbcfa14990fd3dfe5d539602e634`；另保留 8 epoch 对照报告 SHA=`90ef40560f5342842dc379c2f6b422a6e549d270d78aa879b18cf01ed16cd87d`，仅用于显示小数据重复轨迹的拟合诊断，不能解释为跨实现或漏洞利用能力。

### 2026-08-10：PG-388 实际本地前后端 canary 采集

- 新增 `scripts/run_pg388_logic_canary_live.py`（SHA=`e96ec3b3faf2f7221c2dc0cfe4d6e7a6b15baf8777a14797c0cbc00bde57aa2d`）和 `tests/test_pg388_logic_canary_live.py`（SHA=`f6a8fb5ec45c936e88fbda1b1b6e5077e9a6d0ab9034dbb57fa4a7df7b5201b7`）。live 入口要求 `PG388_LOCAL_EVAL=1`，只允许 localhost/127.0.0.1，拒绝外部 URL；未显式开关时不接触前端。
- 在当前 PG-388 Docker 前端/后端上完成真实序列：3 次 fresh reset、15 次 typed observation、4 个 candidate/replay/scope effect、3 个 negative clean、unsafe allow=0。报告=`research/pg388_logic_canary_live_v1.json` SHA=`63ef2785e616e2aac186303b73ffda19910a5dcedee8836b05bb6a6b69bed526`，状态=`passed_live_local_canary_only`。
- 报告只保存抽象 state/effect/action/invariant 桶与逐角色 evidence hash；`target_contacted=false`、`external_network=false`、`wire_created=false`、`safe_to_send=false`、raw request/response 不存储，training/promotion/memory/payload/vulnerability 全关闭。这是本地模拟器的 typed 状态差分，不是任意应用漏洞或可迁移 payload 能力。
- 本轮规则文件 SHA=`100067d7b2823fce75feb16aa3a260398cdd1a52b4374417ccc51459d8fa855c`，演示资产 manifest SHA=`e6d6ccb0c268fcd6d2aac162cffcb85831cb084c3eb1f8b31ddcb7e2b0dac8e1`。

### 2026-08-10：PG-388 扩展本地逻辑 canary 矩阵

- `/api/canary` 已从 3 个状态机扩展为 17 个 enum-only case refs，覆盖安装重入、价格/状态/数量边界、身份规范化、找回绑定、2FA 顺序、验证码重用、Session 旋转、水平/垂直越权、标识符枚举信号、执行顺序和响应投影；其中 14 个有受控 defective-branch effect，仍只返回抽象 state/effect/action 桶。
- 后端实现最新 SHA=`166e6fa7e8785492aa4ac6df4250c024bf1bbe7e60b0ad6721c3a1ec2e8df5a0`，README SHA=`2225dbcb0271b1c647b40936418e479d46f2a9a18f20d49209e1cfbb81f05541`，刷新后的本地 display image=`sha256:80b2a43dab8cd2a0744ca4fc06b94f9948389392272fd1a9842eccc1d6a030da`，容器刷新报告=`research/pg388_logic_backend_container_smoke_v2.json` SHA=`956b295106cb1ed606a479241e8f42363aee5dda7b31fce3cd72dcb3b871e20a`。
- 真实 HTTP 验证了 `install_reentry_gate` candidate 产生 `setup_reentered` typed effect，以及 `two_factor_reset_binding` negative 保持 clean；所有请求仍只含 `case_ref/role/phase`，safe_to_send、外网、持久化和 promotion 全关闭。
- 扩展矩阵登记后的当前 rules SHA=`c53a310fc64bdb52407171f96462b8840fa7ab0c41de166e5aca4bb76e51e5c4`，demo asset manifest SHA=`0136dc90c17e447ea6509827f364f47a7d7a009e689ebd155e20ed5bfe2095fe`。

### 2026-08-10：PG-388 17-case 轨迹集扩展

- canary trajectory builder 现在覆盖 17 个逻辑 case，生成 510 行（train/implementation_holdout=`255/255`）；dataset SHA=`7cefae6b85d380f13cf79c15f1cfd06c51ba30ab0004d776ef8af750d81a1a1f`，builder SHA=`ae85150297601c90a561f98143f31bf4bbb041145615059ca00b0cc5aaed2178`。
- 审计报告 SHA=`f2574f0cfe345ee742c86434f3b0239ba6db52b194b5b14cf4faa9c594d83b68`，审计脚本 SHA=`d70d668c0ed64ec60bbde401741593e182230b9f8e3d8a59c6d0c673d25916c0`；cross-split context/context-target overlap 均为 `0`，raw/context firewall 与 training/promotion gates 仍关闭。
- 新数据上的 1 epoch CPU wiring smoke：train/holdout=`255/255`、unknown context gap=0、optimizer 仅 CPU，报告 SHA=`03210760ff87e80e87afcf9fba85cdf78215df1eae56b766412dde4d854d0e28`；8 epoch 对照 SHA=`80701dd04ca79a3b1fef10a977491b0c79fa467a884b1730e29b5ab66fa630f8`。8 epoch holdout `next_action` 最坏约 `0.9647`、ASK recall=`1.0`，只代表本地小矩阵拟合诊断，不代表通用漏洞能力。
- 该扩展批次最终 rules SHA=`2c0286cbfe23b7671577f51b808ae7013eab099a8e4e512aa85ff5300b0b91d9`，asset manifest SHA=`d961700d92675b026f6bcf51893bf220b081f1424e477e67ee4e899face31d8f`。

### 2026-08-10：PG-388 全矩阵 live replay

- live runner 已与 17-case 矩阵对齐，真实本地 HTTP replay 重新执行 17 次 fresh reset、85 次 typed observation、32 个 candidate/replay effect、17 个 negative clean，unsafe allow=0。报告=`research/pg388_logic_canary_live_v1.json` SHA=`3a4ce6d69673e4a2c5690284174892d8a29bf17a5942cd2b87b2f4b7f288f1c1`。
- runner SHA=`0ceac2d4bf942a15b7ddc6d50c25262993a90a6705e258c51eca4e3a29d25633`，test SHA=`9214b9b1ebfa9419e176857d6bb04e16fa34e38af58e22899125d3071e3d8e0e`。live 仍只允许本地 origin、enum-only body、抽象 projection；raw request/response、外网、wire、持久化、训练和 promotion 全关闭。
- 全矩阵 replay 更新后的 rules SHA=`86eb48a2a08c3de04efb6fc4acc8d22ef9245f8b165cc692bef60b29236c9721`，asset manifest SHA=`66bb4263869aab09a9f811bfe2d2d3873a1c17c16d4623ff12ea0e76512fced5`。

### 2026-08-10：PG-389 JS 解码/过滤链抽象实验与展示

- 新增 `app/pg389_js_chain_projection.py`（SHA=`e4c654dc3e644fb53b6fa2765c3a7c050aef6e281b00886d1cfbd027bbf7780b`），覆盖 12 个有序链案例：query/form/fragment/JSON 解析、单/双层解码、trim/casefold、过滤阶段、guard 优先级、text/attribute/structured/state/code sink 和黑盒 observation sequence。`project_js_chain_source` 只输出抽象 token 与 source hash，不返回脚本、字面量或输入值。
- 新增 builder `scripts/build_pg389_js_chain_dataset.py`（SHA=`2097251509605db4afa796f75cc18029e7862e91a75162d9fe9289e512f2b58d`）和 audit `scripts/audit_pg389_js_chain_dataset.py`（SHA=`5c924aab981cdd5c00345bd9a55915253b007fa6fadd51cc5ae8dfa62a5e17e0`）。数据=`research/pg389_js_decode_filter_chain_dataset_v1.json`（288 行，train/implementation_holdout=`144/144`，12 cases×2 implementations×3 seeds×4 roles，SHA=`03181b72a5eded91f39657aba591698355e2ba775d397458e90b539c96e001b7`）；审计=`research/pg389_js_decode_filter_chain_audit_v1.json`（SHA=`a7cd42ced93879f6b87c44a94ff4c8eb5feb10578e5aee72c08e4007661eb903`），status=`passed_candidate_audit`，cross-split context/context-target overlap=`0/0`，raw marker/row hash failure=`0/0`，training/promotion 全关闭。
- 新增 `/pg389` 前端抽象演示：`frontend/app/pg389/page.tsx`、`frontend/components/pg389-js-chain-lab.tsx`、`frontend/components/pg389-js-chain-lab.module.css`；页面展示 ordered decoder chain、filter stage、guard precedence、sink context、black-box observation 和 ASK/REPAIR/ABSTAIN/SELECT，不展示原始脚本或具体探针。Next build 通过，`http://localhost:3000/pg389` 与 `/pg388` 均 HTTP 200；当前本地 frontend image digest=`sha256:6a462a00f24ff316145ea3211bce7e0fc81c5d8809b7cfbbc763d05462749a68`。
- `tests/test_pg389_js_chain.py`（SHA=`9b03406116f53270c230fe343cbb49a1d644a0054f60d429a877403e4b3fa1f8`）6 项通过；PG-387/PG-388 相关回归与 `scripts/verify_demo_assets.ps1 -AllowMissingOptional` 通过。rules 当前 SHA=`bb2154973d566402d08bab2428be8bd1b89ff06ce950af584ffe9b241f7da95c`，demo asset manifest 当前 SHA=`ccf45d6b5cc6da3d52be1b7622cf7e500110c7821e33a0e07cd7f0210707e749`。
- 该实验只证明抽象 JS 链路覆盖和审计合同；不代表模型能生成任意 XSS/SQL/WAF 绕过字符串，也不产生 live 漏洞确认或 payload catalog 晋级。具体本地 evaluator 如需扩展，仍必须另行授权、fresh reset、正负对照、typed oracle 和 evidence hash。

### 2026-08-10：PG-389 本地 fixture typed replay

- 新增 `scripts/run_pg389_js_chain_local_replay.py`（SHA=`bd9ffc685338a026b4587c0d0d2e84758c8f9a176ff9dacc9acdb21d6364bf86`）和 `tests/test_pg389_js_chain_local_replay.py`（SHA=`0d20bea66448593cd491717b303550db6e3f9767542650038837342a946067ff`）。runner 复用已审阅的 PG-385 loopback inert fixture，把 3 个 PG-389 解码链映射到 candidate/reference/negative/replay；每角色 fresh reset，失败→修复动作变化和证据哈希留存，具体 fixture 绑定字符串只在 evaluator 最后一跳短暂存在。
- 报告=`research/pg389_js_chain_local_replay_v1.json`（SHA=`95e68add2f5d9405d2a3e7e0f52409476bf5222a310df23101c35c217aad6453`）：12 行、12 fresh reset、9 typed effect、12 baseline filtered、12 action changed、negative violation=`0`、evidence=`12`；`local_fixture_contacted=true`、`target_contacted=false`、`external_network=false`、`raw_wire_stored=false`，training/promotion 全关闭。它是 fixture-shape diagnostic，不是独立 JS 实现、任意目标测试或通用漏洞/payload 能力。
- `/pg389` 页面增加 fixture replay 状态条（9/9 typed roles、negative violation 0），前端镜像刷新为 `sha256:bc0ecb261812f6dd0f7004e75cb86d5e0fde6da78463250d3991c51597d60902`；本地 `/pg389` 与 `/pg388` 继续 HTTP 200。当前 rules SHA=`6032036900469ba55c319dc9dd68bd69fa4ede7855f46fecb18b3b9768cc1572`，asset manifest SHA=`013dbef9d5bff05e9b70306b3762cb515aaa26f42942e4ede9637df593a2d57e`。

### 2026-08-10：PG-388 交易并发/幂等锁 canary 扩展

- `/api/canary` 新增 `purchase_concurrency_lock`，把 4.13 交易项的“并发数据库锁处理不当”落成抽象 `transaction_concurrency` / `order_version → one_commit` 状态机；仍只接受 `case_ref/role/phase` 枚举，不实现真实并发线程、订单、支付、数据库或持久化写入。后端 SHA=`47c926a81b48f17716b7a5fc5f9dc740eaa8704d5d4af45c3d2b30d9d4c9f1d5`，README SHA=`32e7547534ea4c2a0f92ba22d7baa75f7080cc1947e9edf9d34dc92f280b40c5`。
- 本轮在固定 digest 的 PG-388 本地 display compose 上重新 live replay：18 次 fresh reset、90 次 typed observation、34 个 candidate/replay effect、18 个 negative clean、unsafe allow=0；报告=`research/pg388_logic_canary_live_v1.json` SHA=`37b04f20680a9ee99cbbed031cdfd68e9508f72fe58b38c0acb134d2e0f02348`。backend image=`sha256:29813d94c47c785dcec86b621b74eed233790c0db707267f6a1b8f493fb96ce3`，frontend image=`sha256:7babb24d23dfcdb96b47025b043e78c0034a7508673662b8f83e992b7a21c3dd`；`/pg388`、`/pg388-api/api/manifest` 均 HTTP 200。
- trajectory builder 现覆盖 18 cases，生成 540 行（train/implementation_holdout=`270/270`）；dataset SHA=`d14d6fc44c273767905aca555a228c2b4508e097463d7bf0deadbda70b8b9649`，builder SHA=`288180272c43edd668843b8b722c7659ff41b69139b052e8b90a1de08604af24`，audit=`passed_candidate_trajectory_audit`，audit report SHA=`318d69af23dea12aab852521560626d4039726a120aa3deca112477b1e7ad7a4`，cross-split overlap/context-target overlap=`0/0`。
- 前端 `/pg388` 增加“订单并发提交”案例，展示 `order_version`、single-commit、candidate/reference/negative/replay 和 `effect_delta_zero`；Next build 通过。该 case 只证明本地模拟器的抽象锁竞争形状，不能宣称真实 race、任意业务漏洞或可迁移 payload。所有 raw/evaluator answer、training、memory、payload catalog、vulnerability promotion 继续关闭。规则文件 SHA=`496bfa0fde07a9cf345616dcc201aeee41c38fe130168b961064dc551a1b4af5`，演示资产 manifest SHA=`2844c33f6a8ebf99aebc4af3a20a147fb0fafab0b6165c8adbe8243efc743ec7`。

### 2026-08-10：PG-388 concrete canary 前端绑定修正

- 审计发现前端交互 runner 旧逻辑只把 3 个 case 绑定到 `/api/canary`，导致已登记的其余 concrete case（包括 `purchase_concurrency_lock`）在 UI 中退化为 abstract-only。现将 18 个 backend canary case 全部纳入 allowlisted 前端集合；10 个 supplemental taxonomy-gap case 仍保持 abstract/ASK，不会被误当 live canary。
- `/pg388` 重新构建并验证 HTTP 200，页面包含“订单并发提交”，前端镜像=`sha256:d813624ca0ab153304c6e95fd733ae52870b3cf5afe9ba8d9a3fa4bbfad5fe9`，component SHA=`0646710c65a3bff701abc3a0c34f81a7f6e08e125d24933bd16f6ebc7c160245`。规则文件 SHA=`d9a5d4d97a10e2fd7b6b370077e7b6a82e8fd53ea8c3ade88a8cf6e14d80ca8f`，演示资产 manifest SHA=`2844c33f6a8ebf99aebc4af3a20a147fb0fafab0b6165c8adbe8243efc743ec7`。

### 2026-08-10：PG-388 supplemental 4.13 canary 扩展

- 将 10 个原 candidate-only 的细项补成 abstract local typed canary：`oauth_second_factor`、`activation_link_second_factor`、`csrf_disable_second_factor`、`captcha_predictability`、`captcha_response_exposure`、`captcha_client_validation`、`captcha_delivery_abuse`、`session_guessing`、`session_forgery`、`session_leakage`。它们只返回状态/效果/动作桶，不执行真实 OAuth、验证码投递、Session 操作、CSRF 或凭据流程；taxonomy audit 的 candidate-only 语义仍保留，表示尚未形成可训练/可晋级的真实来源证据。
- PG388 concrete canary matrix 现为 28 cases、25 abstract effect cases；fresh local replay=`28/140/54/28/0`（resets/typed/candidate-effects/negative-clean/unsafe-allow），报告 SHA=`72064d95e636769029b75dc922c4ed2b18f26b6fa0db7978f880ee47f7796a23`。轨迹数据扩为 840 行（train/implementation_holdout=`420/420`），dataset SHA=`83ee6bbc03df77c753293fac1d2076bddb78987302a359a8a7cb17161d7a6654`，audit SHA=`7441d86aa7f5993b7fac9f63d88c044c6471b52b1ede3c8fd7648f570ac16e4b`，cross-split overlap/context-target overlap=`0/0`。
- 固定 digest backend image=`sha256:03aadf776c1aeb315ba64fff524c0cf53d18ca017c53f007515cd9cbd70abe36`，backend source SHA=`a38be42a2792b0114be393c5e0c9f12dcf7023be390a4b4d874484d50d615a6f`；`/pg388` 页面仍 HTTP 200。所有数据仍 `training_eligible=0`，raw/evaluator answer、training、memory、payload catalog、vulnerability promotion 全部关闭。扩展 28-case token CPU smoke 报告=`research/pg388_logic_canary_token_cpu_smoke_28case_v2.json`，SHA=`b235eea07df8081abd5ed482b326d1764b05290808e3e01b785d1bcdc56cc298`；420/420 train/holdout、train-only vocab=75、unknown gap=0，但 worst ASK recall=`0.0`、logic invariant=`0.042857`、state transition=`0.064286`，仅为 seed-unstable wiring diagnosis，不是能力成绩。规则文件 SHA=`9fa351f4875cd9cbbc318aa1e2768658e71f869c250290ddc441aa8f674d33ac`，asset manifest SHA=`92b264afe7c6ae27c69c6f6d426ec2cc52d3fb1b8c0b7f40370a874cc89d0e26`。

### 2026-08-10：PG-388 逻辑头 pooling 诊断

- `scripts/run_pg388_logic_token_candidate.py` 增加可选 `boundary`、`mean`、`mean_boundary`、`anchor_mean_boundary` pooling；测试 SHA=`052104fa6346ff779443b1662561aa016451eafa04cdcd141a5f2bfab0308aca`，runner SHA=`7a48838f34ee730175243960b6d4a6ba781125c971551330bd657e4369d081df`。
- 同一 840 行、420/420 split、2 epochs、CPU-only 对照：`mean_boundary` 报告=`research/pg388_logic_canary_token_cpu_smoke_28case_mean_boundary_v1.json` SHA=`c1819f50faf87ae6968227b8c0736c6f0712c9107455e7f800951527ae89eef0`；`anchor_mean_boundary` 报告=`research/pg388_logic_canary_token_cpu_smoke_28case_anchor_v1.json` SHA=`362385859ed32bcfeb22633d000a65160645418a07552a19e8c8a0c5487e5760`。
- anchor pooling 将 state-transition 最好 seed 提到 `0.085714`，但 worst ASK recall=`0.0`、logic invariant 最低约 `0.042857`，仍远低于能力门；这说明仅换 pooling 不能解决逻辑不变量组合学习，结果仅作模型结构诊断，训练、payload、memory、vulnerability promotion 全关闭。当前规则 SHA=`23f5192b2d2ae2525bdae06ed9484cf2f1d08a1d32fb0ca157740bcd8ce007fa`，asset manifest SHA=`ecae950c5b642f86fb75869dc6212ee1f5a6bfb82f16aa80df88caa4b7670a0a`。

### 2026-08-10：PG-388 pooling smoke 时间窗隔离修订

- 上述 pooling smoke 发生在 `05:50 Asia/Shanghai`，不满足本地 `08:00–18:00` 训练窗口；新增 `research/pg388_logic_pooling_time_window_quarantine_v1.json`（SHA=`316709b6aa7a710274d2ea7b4883b6b2eefd2a78753b9300715428c913d8fc72`），将两个报告标记为历史诊断、不得训练/晋级。后续只做代码/合同测试，合规时间再决定是否重跑。修订后规则 SHA=`fb9a8dc2781df7c7709188c361d4d860c962b0a89d21d172297b196f7907a02b`，asset manifest SHA=`bef0f86cf38c8a0b4849f5f14999837a48a5c00397174e3840b98139475d633f`。

### 2026-08-10：PG-388 结构化逻辑槽组合计划

- 新增纯标准库计划器 `scripts/plan_pg388_logic_composed_candidate.py`（SHA=`4a8800bbdabafa2287330d551f9aefb4341f4b02800018309b0c7ca87a38a936`）及 tests=`f1342e43db363d832e1a5a06a516124baf973b5d89f506928485e7fd9b67f400`。它定义 11 个有序 Rule-IR 槽和 `autoregressive_causal_previous_slot_conditioned` decoder 设计，目标是让 invariant→transition→action→repair 组合，而不是独立标签记忆。
- 计划报告=`research/pg388_logic_composed_candidate_plan_v1.json` SHA=`2097845dcd68c696863c7f091a17df9a712401c0aa4e87f0aa7b4d26bc5b0595`，840 行（420/420），状态=`blocked_capability_contract`；typed evaluator、fresh role reset、operator review 均未具备，optimizer/GPU/Docker/network/wire 全部 false，training/promotion 全关闭。当前时间窗外只完成 plan/contract，不运行训练。当前规则 SHA=`d8ef65e4bd79dfa504ff4bc552a8c3cb8cc3178423a680248de13cc782dec08b`，asset manifest SHA=`4518393b26b08fd43bbc41d2556298e6652eb7e417c23e46252d46d961cff1c7`。

### 2026-08-10：PG-388 模型门演示面板

- `/pg388` 新增 `MODEL READINESS / STRUCTURED RULE-IR` 面板，明确显示 11-slot previous-slot decoder 设计、840 条 `train/implementation_holdout=420/420` 抽象轨迹、当前 wiring 诊断和 optimizer 前硬门；component SHA=`4dd6185a9f770225111ae5b7da0f134a66b40149e16bf3996ec48323b73e3964`，本地 frontend image=`sha256:c92593c232745a2de1ff733650c0331b9966f81008c9487e4857b6ac7e3083e2`。
- 页面与 `/pg388-api/api/manifest` 已复核 HTTP 200；manifest 资产数=`27`。面板明确 `optimizer=0`、typed evaluator/fresh role reset/operator review 未认证；训练、payload、memory、promotion 和 vulnerability claim 继续全部关闭。本次提交前校验：rules SHA=`1b5e379b14a3e157928d8f8dff6d1dbba65b92d8731e61eb2c31bb1a9d3a7d3f`，asset manifest SHA=`06de4d755de62f685a606c7f69aa90c4cd618f1b9f051421b849228dd2e616a4`。

### 2026-08-10：PG-388 结构化 Rule-IR 组合数据绑定

- 新增 `scripts/build_pg388_logic_rule_ir_composition_dataset.py` 与只读审计 `scripts/audit_pg388_logic_rule_ir_composition_dataset.py`；从 840 条 trajectory 生成 11 个有序 slot 的模型视图，`effect_shape/state_delta/invariant_result` 保持 evaluator-side summary，不进入 `target_tokens`。数据=`research/pg388_logic_rule_ir_composition_dataset_v1.json`，840 行（train/holdout=`420/420`），audit=`passed_candidate_rule_ir_audit`，unique row hashes=`840`，context firewall 通过。
- 本地 live canary 只以 aggregate coverage 绑定（fresh reset=`28`、typed observation=`140`、negative clean=`28`、unsafe allow=`0`），没有 row-bound implementation/seed/evidence 绑定；因此 `training_eligible=0`、optimizer/GPU/Docker/network/wire 全部关闭，不能把该数据称为漏洞能力或 payload 数据。
- 本次资产校验：builder SHA=`fad11aa4e3b9d95bc66f5498c758d9f73492df90c02f204a3b42046e524a9f1e`，dataset SHA=`ae9461fa84062ab2d9a96dbd3450ff83932e4f2f79d7baa7ae51703faeccef26`，audit report SHA=`45463e834e71394831fd60c37cc52edebd5057abb069408a2e14163e20d0867f`；规则 SHA=`783e8f012ce50f314691a7d16790a5ea5a62c31ef2e1e58d9d10df7769e9836b`，asset manifest SHA=`d22c1a1d7fbbd1e6659e5a38da4703c27a7fc961431ff724a1b2f768f529dc2b`。

### 2026-08-10：PG-379 独立实现动态整页 source-row fresh matrix

- 修复 PG377 JS overlay 对有界 `script_count` 的白名单遗漏，并修复 JSON/文本响应与 HTML fragment 的 `doctype/html_lang=absent` 字段语义；相关 adapter/test 回归通过（当前 `tests/test_pg331_vulnerableapp_adapter.py`、`tests/test_pg377_webgoat_source_row_adapter.py` 等组合 40 passed）。
- 在两个独立、固定本地镜像上完成 1 seed × 12 GET/POST route classes × candidate/reference/negative/replay 的 fresh matrix：96/96 role episodes、72/72 strict PG-331 source rows、72/72 typed roles、24/24 failure→repair、negative violation=0、fresh/reset/network-none/loopback/13-slot/context-firewall 全通过；无残留 pg379 容器，GPU/training/network external 全为 false。
- 最终报告=`research/pg379_dynamic_source_rows_live_matrix_v5_20260810_report.json` SHA=`0d918a25a20f598452ef605fb90e53bcff135dbb9aa6509afefee76ae03da754`；抽象 rows SHA=`524d6be0759e656db2835ce305798d664d60eca445bd62b143c24065dfcac8b9`；sidecars SHA=`b5fc580376d3510b9f6071f1594900337003a0fe204371cadad3875de885112b`；只读 audit=`research/pg379_dynamic_source_rows_live_matrix_v5_20260810_audit.json` SHA=`12358b55515c141b8cc14f831c0085d59e9291f8c6f29f85994240735f4b696d`，`strict_valid_records=72`、`authorization_matches=72`。
- 采集器=`scripts/run_pg379_dynamic_source_rows_live.py` SHA=`9b7696164e1e07e6834131f43b4113e7d818b3b3d1247318a477d11f1514ff84`；audit=`scripts/audit_pg379_dynamic_source_rows_live.py` SHA=`28116ef9cfcd618ec152ff3856cd053d1f4aa544321905c03a8a29b71369b256`。该轮只作独立实现 candidate/evaluator evidence；`training_eligible=0`，不能回填 PG388 逻辑轨迹、不能宣称通用逻辑漏洞或 payload 能力。

### 2026-08-10：PG-388 Rule-IR audit 投影接入前端

- 新增只读投影器 `scripts/project_pg388_logic_frontend_summary.py` 与测试；它从 PG-388 composition dataset/audit/plan/live report 读取计数、槽位、哈希和 hard-gate，只输出有限摘要，不复制 rows、context/target token、payload、wire、response 或 evaluator answer。
- `/pg388` 的 MODEL READINESS 面板现在读取 `frontend/public/research/pg388_logic_rule_ir_frontend_summary_v1.json`，展示 840 行、420/420 split、11 slots、audit `passed_candidate_rule_ir_audit`，并明确 typed row binding、fresh role reset、operator review、training/promotion 仍为 HOLD。
- 当前投影摘要 SHA=`e4d027edc023ff02b8d195bf0b3b77033cdd2d1d4e5e78012f449cb9e18645a3`；前端组件 SHA=`65b0a309dc65d426a5971c883ec53cdeafd646bd9a4cece2218afbd514d40cc2`；样式 SHA=`558818e85b6b2d0ff183750c404f673a4bb5be1e1c1074b241915a14d841cf1b`；demo asset manifest SHA=`ac60be864ee644909e4ad45f69740ef9e53936c596227faa220c9eb77e5a30b1`。Next production build 与 20 项 PG388/389 回归通过；不代表逻辑漏洞泛化或 payload 能力。

### 2026-08-10：PG-388 row-bound Logic Rule-IR source-row matrix

- 新增本地只读采集器 `scripts/run_pg388_logic_rule_ir_source_rows_live.py`（SHA=`042cc273796c49814cae4a7b8bebaba17976faf16303277e8dfbbd62997beafa`）与审计器 `scripts/audit_pg388_logic_rule_ir_source_rows.py`（SHA=`448e27d05ff2f249c16cf9c6fd68159d77e375e31fbd6d46338e2c1ebeccd175`）。采集器只接受显式 `PG388_LOCAL_EVAL=1` 和 loopback `/pg388`，不启动 Docker、不访问外网、不写 wire/payload/响应正文；页面只在内存中解析为 PG-331 七轴/107-field manifest。
- 28 个逻辑 case × 5 个角色/阶段（candidate/reference/negative/replay/failure-repair）完成 `140/140` row-bound episodes：source rows=`140`、strict valid=`140`、typed observations=`140`、fresh role resets=`140`、failure-repair=`52`、negative violations=`0`、raw literal hits=`0`、unique record refs=`140`。Rule-IR 目标固定为 11 个抽象槽（含 `question/ask_reason/invariant/transition/precondition/counterfactual/probe/next_action/repair/oracle/safe_to_send`），具体 wire、答案和原始字符串仍仅 evaluator-side。
- 产物：report=`research/pg388_logic_rule_ir_source_rows_live_v1.json` SHA=`fdefa0be1292a154cff93d4a142f0b63fbacd1c15e22b2bd4bfbdec8c65e3567`；rows SHA=`642df7b4abf79789a4ca581767190832d0fba3c1728bb9bc8f9d2ba5336dcc1f`；sidecars SHA=`8234cc5362ec7414731571f369d01cc22fcf687f0693557f30c61327389c84a1`；audit=`research/pg388_logic_rule_ir_source_rows_live_audit_v1.json` SHA=`b287bf46d6ea44adb4ef25e65e80e0d76f6dfcba3d56625ec55dc650d06cc75c`，status=`passed_candidate_logic_rule_ir_source_row_audit`。
- 这是一实现的本地前端 candidate-only 证据：`image_attested=false`、`operator_reviewed=false`、`split=implementation_holdout`、`training_eligible=0`；fresh/typed/negative/row-bound gates 通过只表示采集合同完整，不表示通用逻辑漏洞或 payload 能力。前端投影器已优先读取 row-bound report，面板显示 typed binding/fresh reset PASS，同时保留 operator review/training HOLD。投影脚本 SHA=`f401e4513655364330add32cc16ecacab0e249f7424162654a88fe0ed4c7b04f`；摘要 SHA=`46e1bcea689dd3bf7840b0c3583104889a7cb1358bafcc72adc5e5fce4f80b8c`。
- 该 row-bound 合同已登记到 `research/improvement_rules.json` 的 `pg388_logic_rule_ir_source_rows_matrix`；当前 rules SHA=`b5f74d191e4f00b41f3a3e3b99501f6bd93a47f8cb1928a7902773e7b0cb9ca1`。规则只允许本地 loopback candidate-only 证据，未打开训练、长期记忆、payload catalog 或漏洞声明。

### 2026-08-10：PG-388 独立逻辑实现 B（静态 holdout 合同）

- 新增 `fixtures/pg388/logic_lab_b.py`（SHA=`102270024d7e7f40c4ca4a9435d701438228db30ef84b79eeba780873978fb8f`）。B 使用独立的 transition/effect 表和实现模块，但继续只接受 `case_ref/role/phase` 枚举；`safe_to_send=false`、无任意值、无持久化、无外网、无真实业务写入。
- 新增可选 `fixtures/pg388/Dockerfile.b`（SHA=`cc16b097bcca6da455d7e2e45366299b1742b6ec248eebfb77d91ace617aeef6`）和 compose `holdout` profile（SHA=`88455b552eefffd4b1ed87eb88a8dcfe003f5df63fa5f702403d0ce0d153c273`）。默认展示仍只启动 A；B 必须单独审阅 immutable digest、fresh reset、candidate/reference/negative/replay 和 source-row audit 后才能进入 holdout。
- B 静态 WSGI 合同测试 `tests/test_pg388_logic_lab_b.py` SHA=`832bc0351a55bbf4f930c80e5d9b8cdca8b16dccc477745066287d2a842f23a6`，与部署合同回归共 `13 passed`。B 的 image 仍未 attested；进程内 source rows 另有单独报告，不能把静态差异表伪装成 Docker live 或训练样本。更新后 rules SHA=`adadba5ef706b09dcb88974e8dbc6b443d7dd544c68a266f5fa5f439bb570a8a`，asset manifest SHA=`c09a0ac834151a95db3df3718a4620c3b34750ac1e1805fcdaaf936321e1636d`。
- B 随后通过同一 PG-331 adapter 做了进程内 holdout 采集：`scripts/run_pg388_logic_holdout_b_source_rows.py` SHA=`f057ae40aeb97482f77e39ef499290684bf1e52c7c8ac15801a997e32b8312da`，测试 SHA=`832bc0351a55bbf4f930c80e5d9b8cdca8b16dccc477745066287d2a842f23a6`。报告/rows/sidecars/audit 分别 SHA=`103784e82feef9d6d770c9c8d27655a443ecf15e868f45295e1a1ac47d5013e6`/`ec8d3a59530096f96bd11c9296e2358b30ea3da0ecddaa8f3ec33fcf5248dcee`/`6252d111f70daafd7e4fb454950acb8bdc09f5d20abbd57ca474dfbaebbe0978`/`2baec352e100c0ce05e9a622674a367ee7689e5085c4f81946fa2401d055c3cc`；`140/140` strict rows、typed、fresh reset，failure-repair=`56`，negative violation=`0`，audit=`passed_candidate_logic_rule_ir_source_row_audit`。
- 该 B 结果是 `local_fixture_in_process=true`，不是 Docker/image-attested live evidence；`image_attested=false`、`operator_reviewed=false`、`training_eligible=0`。规则更新后 rules SHA=`404e1891385461a40c3b50eed7b389b74a8dfb5751a7c51f1055fa36d42f5aed`，asset manifest SHA=`075bfc0d1ecb812c73413fa568c048ef1e34a7debe50fcf92303b424c5d06787`。
- 新增只读 Docker 预检 `scripts/preflight_pg388_holdout_docker.py`（SHA=`917e59275e0c8d1478d7d1320ab87c200e2250a5f728ddb2608115028bb08d4f`）与测试（SHA=`db0a685cb0a2ebae405b945179965f5b1041bc7d3667ad098916ed3dfcbb0948`）。默认无 `PG388_LOCAL_DOCKER_EVAL=1` 或 immutable digest 时返回 `blocked_holdout_docker_preflight`；即使预检通过也只允许 `docker compose config --quiet`，禁止 `up/build/run`。当前运行结果 `docker_started=false`、`image_attested=false`、`training_eligible=0`。最新 rules SHA=`109dd119c4520186a2db106471c82d510433631d6a8d0b5f45b12df3fbfebf48`，asset manifest SHA=`bed38e4ee9366166ff44ac01bc412a951af8f266aa92a69b54cada912628882c`。

### 2026-08-10：PG-388 独立实现 B disposable Docker smoke

- 在显式 `PG388_LOCAL_DOCKER_EVAL=1`、已审阅 Python base digest、`--network=none`、无端口/挂载、只读 rootfs+tmpfs 下构建并启动唯一容器 `pg388-holdout-b-smoke-20260810`；容器完成后已 stop/rm，并核验不存在残留。image=`sha256:69f4e356904d0be168d98f4492b4c388b3caa0a55093b784789dd90a4bb4ac9e`，base=`python:3.11-slim@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff`，B source SHA=`102270024d7e7f40c4ca4a9435d701438228db30ef84b79eeba780873978fb8f`。
- Docker smoke 只执行抽象 health/case/canary/reset：health HTTP 200、56 cases；candidate/replay 为 `bounded_transition`，reference/negative 为 `zero`，negative clean=true；4 个角色均有 fresh reset before/after，external network=false、persistent storage=false、safe_to_send=false。未保存原始输入/响应，未启动训练/GPU；报告=`research/pg388_logic_holdout_b_docker_smoke_v1.json` SHA=`9a846e90d6a209322746088b78168aae31727c1907c1ff5b1d1dee3e86d50cc8`。
- 这是一次本地 evaluator-only 运行证据，不是模型能力、通用逻辑漏洞、payload 或晋级授权；`training_eligible=0`、training/memory/payload_catalog/vulnerability promotion 全部关闭。asset manifest 已追加该报告，当前 manifest SHA=`a18ad666c6d76cb3581e2b568201e89d9f801a805f70d9d8813976fc6c4699b8`，规则 SHA 保持=`109dd119c4520186a2db106471c82d510433631d6a8d0b5f45b12df3fbfebf48`。

### 2026-08-10：PG-388 前端展示接入独立 B / Docker projection

- 只读投影器 `scripts/project_pg388_logic_frontend_summary.py` 已接入 B 的 source-row audit 与 disposable Docker smoke；摘要 `frontend/public/research/pg388_logic_rule_ir_frontend_summary_v1.json` 现在只展示 B 的 bounded counts/status/fresh/negative/docker gates，不复制 rows、tokens、wire、响应体或 evaluator answer。当前脚本 SHA=`2beda288adcc3805cc94c373b6223f48c57984aec5d8b39843374ecbbc7f668c`，摘要 SHA=`3a21ae03dde7ca3e4d60d12caa6c70e50650955736e8c0cd5beaef182ed9c683`。
- `/pg388` 的 MODEL READINESS 面板加入“INDEPENDENT IMPLEMENTATION HOLDOUT”：B source rows=`140/140` strict、typed=`140`、audit passed；Docker smoke health=`200`、4 roles fresh before/after、negative clean；image/operator review 与 training 仍 HOLD。组件 SHA=`03b99b5ca02f3f2526bb75f0f71019a057cf0c4c25c5b77869adadbad9b7504f`，样式 SHA=`1f752bad42d27890518c70be827225133ddd0161b3c310843eb8f5a061ccaec0`，projection tests SHA=`5bcd8d02dc5edd25ebe4a415465f140a131254ee2e8906692e018e44122d243c`；Next production build 通过。
- 资产清单当前 SHA=`3c9224ba5ce5683438b424b76a11d176612785a982da3c964f6f309af3223911`，校验 `asset_count=60` 通过。标准 Dockerfile 的 `npm ci` 在 `network=none` 下超时；随后使用刚通过本地 `npm run build` 的 `.next/standalone`，以原前端镜像为底层做无外网 refresh image，并按原 `pg388demo` project 仅重建前端服务。当前前端 image=`sha256:5673b3938e6d48c4c0a5b10151703478d7ae819f4dfbbf39123a74d777e37696`，`/pg388`、投影 JSON、`/pg388-api/health` 均 HTTP 200；B holdout 服务未启动，后端未重启。
- 新增跨实现只读审计 `scripts/audit_pg388_logic_cross_implementation.py` SHA=`9590d1a536ce767d1eb1d2bbef63c4af1e98eb3bbad61c17cd8b86e498974309`，报告=`research/pg388_logic_cross_implementation_audit_v1.json` SHA=`bc3dd567f7028bd1efc1ce752fef9bdc447a499a994f7c76fbace8952eab3aa0`。A/B 合计 `280` 条 source-row wrapper、strict/typed/fresh=`280/280/280`、negative violation=`0`、context overlap=`0`；但两者全是 `implementation_holdout`，`train_split_missing`、operator review/image attestation 缺失，`training_eligible=0`，所以只能作为 candidate/evaluator 证据，不能进入模型训练或晋级。

### 2026-08-10：PG-388 候选模型指标接入前端投影

- `/pg388` 的模型面板新增 `CANDIDATE MODEL / CPU SMOKE` 聚合区：只读取三份既有 CPU candidate 报告的 train/holdout 数量、train-only vocabulary scope、seed 数、最弱 Rule-IR head、ASK worst recall 和 negative false-allow；不复制 token 序列、原始 payload、wire、响应或 evaluator answer。
- 当前三份报告均为 `cpu_smoke_candidate_only`、device=`cpu`、GPU/Docker/network/wire 全 false、`training_eligible=0`、promotion 全关闭。logic invariant smoke 的 worst ASK=`1.0`、negative false-allow=`0`、最弱 head=`ask_reason 0.75`；supplemental smoke worst next_action=`0.679688`；trajectory canary worst ASK=`0.0`、logic invariant=`0.042857`，后者明确显示模型 wiring/组合仍不稳定。
- 投影脚本最新 SHA=`b28f469bbc4aae555c5bf6d4110298fd7f5011344ab47a25b925644515321857`，组件 SHA=`2305ef9f9cdaf8c2d681a9a824faa9d79c53c545d49b15a3bcf15a916b2e0950`，测试 SHA=`a7c767419870eb54460f8e9a37b85f94e47fa5519c5acb070fd36e0a078c9670`，摘要 SHA=`05fc5837df2c8ce84f1429239187c74364a80701f06ea57b5650f059fcacc6a7`；asset manifest 当前 SHA=`3c9224ba5ce5683438b424b76a11d176612785a982da3c964f6f309af3223911`，校验 `asset_count=60` 通过。Next build 和前端投影测试均通过，当前展示仍只说明逻辑推理诊断，不代表通用漏洞或 payload 能力。

### 2026-08-10：PG-388 候选模型面板发布

- 提交=`80290d56e47394edc098fb2675dfdcba8542b7e3`，已推送并核对 `origin/main`；工作区干净。
- 发布内容包括三份 CPU candidate smoke 的只读聚合指标、PG388 模型面板、摘要/资产清单同步；回归 `12 passed`、`npm run build` 通过、资产校验 `asset_count=60` 通过、`/pg388` 与摘要接口 HTTP 200。
- 本次提交仍没有启动 A800、Docker holdout 或外网；训练资格、payload catalog、长期记忆和漏洞声明全部关闭。下一步若要真实训练，仍必须先补 train split、operator review、image attestation 和信息/容量硬门。

### 2026-08-10：PG-388 11-slot 组合 decoder candidate smoke

- 新增 `scripts/run_pg388_logic_composed_candidate.py`（当前 SHA=`ea3c372a2a445ff457d5e32d103cb15ffb97361075b83491a4bc8e0d2c6dc799`）和测试（当前 SHA=`643238625eae7bb34accc1ee6d932a180195e9cfe68c53f8468deed3ad8729a2`）。它只读取 PG388 abstract context/11-slot target，使用 shared causal MoE + previous-slot-conditioned composition decoder；evaluator summary、source metadata、wire、响应和 payload 不进入 batch。新增本机工作日 CUDA candidate 门：必须显式 `BLACKBOX_LOCAL_MORNING_TRAIN=1`、`CUDA_VISIBLE_DEVICES=0`、周一至周五 08:00–18:00，且只标 candidate-only。
- plan=`research/pg388_logic_composed_candidate_plan_v1.json` SHA=`8679194a9a080f0d633d3596b5feaa2926bc3453cb46e4f1f3efadd5c8d85566`，source contract 仍因 row-bound typed evidence、fresh reset attestation、operator review 缺失而 blocked。CPU smoke=`research/pg388_logic_composed_candidate_cpu_smoke_v1.json` SHA=`bfc5eea536784979e045496a0323877a272c1ff41e362534dfb6582da0ab182b`，128/128 bounded rows、3 seeds、optimizer 仅 CPU、GPU/Docker/network/wire 全 false、training/promotion 全关。
- 组合 smoke 的 worst holdout composition exact=`0.023438`、slot accuracy=`0.628551`、ASK=`0.705882`，显示“把不变量、动作和修复按顺序组合”仍是模型瓶颈；该报告只作结构诊断，不是逻辑漏洞利用结果。前端 candidate model projection 已纳入该第四组 run，摘要 SHA=`e92ddb32ef427160074588a98cdcc57deb08688e50128ddcc11efd939e332256`，投影脚本 SHA=`7b045e65f7e5a34ed20d57142bc6b89b7a6b9c505933f382aebbedd9e59f251b`。
- 追加完整 420/420 train/implementation-holdout 的 3-seed CPU run：report SHA=`bf8a3cd1d9ff45c762d62fc8381fc9ab4f3e20c63acea3540de8970ab541a31c`；worst composition exact=`0.007143`、slot accuracy=`0.800649`、ASK=`1.0`、repair=`1.0`、negative false-allow=`0`。完整数据没有解决组合精确率瓶颈，仍是 candidate-only。
- 追加同一全量数据的 8-epoch CPU 对照：report SHA=`7e929aa1a95bc4154aa08b81fd6690cfbf8bb34b78a2b3d04372e72f7467327e`；worst composition exact=`0.964286`、slot accuracy=`0.994156`、ASK=`1.0`、repair=`1.0`、negative false-allow=`0`。这说明早期低分主要是训练不足，但仍只证明抽象 Rule-IR 组合，不证明真实漏洞能力。
- 本机 RTX 3060 工作日 CUDA e8 candidate 已完成：`research/pg388_logic_composed_candidate_local_cuda_e8_v1.json` SHA=`b1411726420a32e689843560c697b972bfca6061601aee3deae959ef954dd02d`，420/420、3 seeds、device=`cuda:0`；worst holdout composition exact=`0.992857`、slot accuracy=`0.998701`、ASK=`1.0`、repair=`1.0`、negative false-allow=`0`、composition entropy max=`0.037840`。execution 仅本机 optimizer，Docker/network/wire 全 false，`training_eligible=0`、capability/promotion 全关闭；这是抽象 Rule-IR 组合候选，不是通用逻辑漏洞或 payload 结果。
- 前端 candidate panel 已加入 local CUDA e8 行；最新摘要 SHA=`17c3e010bf174dc9f8fbe4054407a204e321c943eefcdba1dfc2fb1d04a57635`、投影脚本 SHA=`d9aba896facc53b384cb49b4cf8266d6c3d0b824935fdc8258245b372e4e2599`；最新 candidate=`11-slot composition (local CUDA e8)`。demo asset manifest 当前 SHA=`0830784bc18e2e46aea582c1a93e4ba3743c23eff8fbdb4f287751b438a567d1`、`asset_count=72`；相关 PG388/PG389/前端/组合回归已在本次收口复核。
- 前端已用最新 `.next/standalone` 做无外网 refresh image，当前 image=`sha256:709877dca0271d14f3fa507f618cfa5853cb5aadb4f5ba27d30d442078918d15`；`/pg388`、`/pg389` 和摘要接口复核 HTTP 200，PG389 摘要包含 local CUDA candidate，B holdout 未启动。
- 前端摘要新增 taxonomy coverage projection：66 个抽象案例、10 个 04.13 类别、0 个缺失锚点、10 个 candidate-only 细项；只显示有界计数，原始行/业务值不进入浏览器。
- 前端运行时将后端 56 个核心合同与 10 个 supplemental contracts 合并进 case catalog，静态详解优先，其余只使用 ASK/抽象模板；PG388 taxonomy/模型指标摘要 SHA=`17c3e010bf174dc9f8fbe4054407a204e321c943eefcdba1dfc2fb1d04a57635`、PG389 JS-chain 摘要 SHA=`58a570262e9daea700bc1aec81625e633785f92c929c42eb72bbb49b27fbfd99`；Next build 通过，资产校验为 `asset_count=72`。

### 2026-08-10：PG-389 抽象 JS 解码/过滤链 candidate

- 新增 `scripts/run_pg389_js_chain_candidate.py`（SHA=`daf1c518119c570f87b3a1fed4a147451f3e767166fe7dd12dc370ee0661de77`）和测试（SHA=`79b883174722c648d5369a3d950f72d81c3c11ece2273841ed1ad6d39f7ad08d`）。runner 使用 train-only context vocabulary 和 6-slot causal composition decoder，输入仅包含 decoder/filter/guard/sink/observation 抽象 token。
- 工作日显式 `BLACKBOX_LOCAL_MORNING_TRAIN=1` + `CUDA_VISIBLE_DEVICES=0` 的本机 RTX 3060 e8 candidate 已完成：report=`research/pg389_js_chain_candidate_local_cuda_e8_v1.json` SHA=`ff8cad255bf6bb7c20d61a1ab7b81f3d824f1057d24b15a6707362b7532a97b2`，144/144、3 seeds；worst holdout composition exact=`1.0`、slot accuracy=`1.0`、ASK=`1.0`、repair=`1.0`、negative false-allow=`0`、composition entropy max=`0.024376`。
- dataset audit 虽通过抽象信息审计，但 source contract 仍 blocked（fresh/reset、candidate/reference/negative/replay typed evidence、operator review 缺失且有 1 个 train-only context gap）；因此 `training_eligible=0`、capability/promotion 全关闭，结果仅说明抽象 JS 链语义候选，不说明源码理解、通用 WAF 绕过或原始 payload 生成。
- `/pg389` 新增 bounded candidate panel，摘要=`frontend/public/research/pg389_js_chain_frontend_summary_v1.json` SHA=`58a570262e9daea700bc1aec81625e633785f92c929c42eb72bbb49b27fbfd99`，投影脚本 SHA=`c5a2535bf8cb1874d77d2fda43c4fb9cdd3d76d78937768c5301a7b832c1e496`，projection tests SHA=`eabb857042a6731e0ea168ad0b67c4ad9b52d781c1f27e7e1aea6313cebdcaa9`；`/pg389` 页面与摘要 HTTP 200。

### 2026-08-10：PG-388 明日演示运行手册

- 新增 `docs/PG388_DEMO_RUNBOOK.md`，记录克隆/资产校验、immutable base digest、compose 启动、HTTP 健康检查、推荐讲解顺序、模型指标解读和安全清理命令。
- 手册明确 PG388 是本地 disposable 抽象逻辑状态靶场；不会把 CPU smoke、typed state-shape 或 holdout 误述为通用漏洞、XSS/SQL/WAF 绕过或 payload 能力。

### 2026-08-10：PG-388 supplemental 4.13 fresh-role canary replay

- 新增本地纯内存 evaluator runner `scripts/run_pg388_logic_supplement_canary_local.py`（SHA=`46251a5ee77012500e88530fd840f7e36adf79fa76119f86bbe731afc3fff3f3`）及只读审计器 `scripts/audit_pg388_logic_supplement_canary_local.py`（SHA=`879dcd02382d01dccc3a60b54c2f126d5bffa9996e10f5be255fad321835da8a`）。runner 固定调用已审阅 `fixtures/pg388/logic_lab.application`，不打开 socket、不启动 Docker、不接触外网、不接受任意值；只保留抽象状态/效果/动作桶和 role-bound evidence hash。
- 10 个 taxonomy-gap case（OAuth/activation/CSRF 2FA、4 个验证码、3 个 Session）× 3 个逻辑 seed × candidate/reference/negative/replay = `120` role rows；每 role 前后 fresh reset=`120/120`，replay evaluator setup=`60`，typed=`120`，candidate effect=`30`，replay effect=`30`，negative clean=`30`，negative violation=`0`，unsafe allow=`0`。报告=`research/pg388_logic_supplement_canary_local_v1.json`（SHA=`eb1302ebdba9f3592cb9e7fed4b2860da8cac5e13f46b0d4882cfeca05ccd630`），audit=`research/pg388_logic_supplement_canary_local_audit_v1.json`（SHA=`5cdc8c729d855cd8d98e30fb570ab8b894a29e539a2273005515f4340289660d`），status=`passed_candidate_only`。
- 报告明确 `in_process_only=true`、`docker_started=false`、`target_contacted=false`、`external_network=false`、`wire_created=false`、`persistent_storage=false`、`training_eligible=0`；它只是 04.13 抽象状态差分诊断，不是通用逻辑漏洞、真实 OAuth/验证码/Session 测试、payload 或漏洞晋级。
- PG388 前端摘要新增 bounded `supplemental_canary` projection 和 fresh-role card，只显示 counts/status/gates，不输出 rows/context/target/evaluator answer；摘要 SHA=`b53abdf969d9275857ace813730970a10ae73f90cca926f1db33dc90bc19ea60`，投影脚本 SHA=`853f37a050d993ea27dce6e00cfcc639bdeacbad62ebd6e6ca489ffb1eccd735`，组件 SHA=`240322d71f509deb11a21960021e88a572d0b384caae247c13b6f686614b2e63`。
- 新增 tests：`tests/test_pg388_logic_supplement_canary_local.py`（SHA=`a2b0c833f3440d2e9b039e6fb0dfb57b618b998383ae87f0d089607bbc2a0d7b`）和 `tests/test_pg388_logic_supplement_canary_audit.py`（SHA=`c500dbeb74d01d8469dc60528f20ca1e626320a95193a2dd9d90a7cbc5835e2d`）。supplement runner/audit/frontend focused=`26 passed`（含原有 PG388 canary/frontend tests），Next build 通过，demo asset manifest 校验=`asset_count=78`。
- 当前演示前端离线刷新 image=`sha256:e88a2427d5bdc64e4d9f236b08e7b64095666398a5a4b8cf886f12905fe682d1`；资产清单 SHA=`3ac7dff3f9d1358e0b650662bc3a8fd5d502c776d77c4eceac6b81fc449fc25d`。标准 Dockerfile 的依赖阶段在 `network=none` 下会因 `npm ci` 无网络而阻塞，本轮使用已验证本地 `.next/standalone` 刷新，不改变默认 compose 合同。

### 2026-08-10：PG-388 typed Rule-IR diagnostic projection

- 新增 `scripts/build_pg388_logic_supplement_typed_projection.py`（SHA=`baa2b0f25dc8fa12e0e70e060b918c3c8dffbdd580b0e55c5929728913d794e2`）和只读审计 `scripts/audit_pg388_logic_supplement_typed_projection.py`（SHA=`519acf85f3c1c51a8150b52883ccde1b6558721cf5a8fd0bef9de44adb47269f`）。它把已审计的 120 条 fresh-role canary 观测投影为抽象 context/Rule-IR target，仅保留 evidence hash/reset/typed 标记；不复制 `effect_shape/state_before/state_after`、raw response、wire、payload 或 evaluator answer。
- 产物=`research/pg388_logic_supplement_typed_rule_ir_projection_v1.json`（SHA=`ce562481a04ff8349e09d4162208a9497c6471f2fb7c437a3326cd7baebdb2f7`）和 audit=`research/pg388_logic_supplement_typed_rule_ir_projection_audit_v1.json`（SHA=`92bbd2c854ee60e8b288236c52300c5f6f1ba7c6d5378c2489dc7c55bf3e0004`），`records=120`、`evaluator_diagnostic=120`、`train=0`、`typed/fresh/evidence=120/120/120`，audit=`passed_diagnostic_only`。`training_eligible=0`、representation candidate 和所有 promotion 全部关闭。
- `/pg388` 新增 `TYPED RULE-IR PROJECTION` 卡片，明确展示“typed observation ≠ train permission”；前端摘要只输出 counts/status/gates，不输出行或 tokens。摘要 SHA=`1f8d9bb8f54192bce7c869b906a15b1a76c3d715c7d7fcc784ef564fc2f1529d`，组件 SHA=`6adac5b27343b6eaa43b31984fddd15db9e7069140bb7f8233646538c007461c`，资产清单 SHA=`e4ac24e135489f34580f91069bd3cdd903108b580ecb9daf312604753d52fdbf`，asset_count=`83`。
- typed 版本已用本地 `.next/standalone` 在 `network=none` 下刷新前端容器，当前 image=`sha256:d889c763006771cd16a4134e801e053f03997f3f01d6e515b3861fb05c8e9de1`；运行态 `/pg388`、`/pg388-api/health`、`/pg388-api/api/cases`、`/pg388-api/api/supplemental-cases` 均返回 HTTP 200。摘要 rewrite `/api/research/...` 不属于 PG388 display API，不能作为前端靶场健康检查。
- 新增 `tests/test_pg388_logic_supplement_typed_projection.py`（SHA=`91bc4798ec4ebb3907789b69170537c9f6ff6c79e5cb2613db98e2fa1ec06b5e`）；typed projection/前端 focused=`5 passed`，Next build 和资产校验通过。该数据没有 train split，不能直接用于 A800 或本机训练。
