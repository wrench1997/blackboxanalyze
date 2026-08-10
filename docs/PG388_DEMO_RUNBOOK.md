# PG-388 逻辑漏洞实验演示手册

这份手册给明天的演示者使用。PG-388 是授权本地 disposable 逻辑状态靶场：页面展示安装、交易、账户、2FA、验证码、Session、越权、枚举和执行顺序等不变量，模型面板展示抽象 Rule-IR 诊断。它不生成任意目标请求，也不把原始 payload、wire、响应正文或 evaluator 答案放进浏览器。

## 1. 克隆与资产校验

```powershell
git clone https://github.com/wrench1997/blackboxanalyze.git
cd blackboxanalyze
powershell -ExecutionPolicy Bypass -File scripts/verify_demo_assets.ps1 `
  -ManifestPath research/pg388_demo_asset_manifest_v1.json `
  -Root . -AllowMissingOptional
```

校验失败时不要启动靶场。`-AllowMissingOptional` 只允许缺少可选 checkpoint；源码、报告和演示摘要必须通过哈希校验。

## 2. 启动本地前后端

使用已审阅的 immutable base digest；不要使用浮动 `latest`：

```powershell
$env:PG388_PYTHON_IMAGE_DIGEST='sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff'
$env:PG388_NODE_BASE_IMAGE='node:20.11.1-alpine3.19@sha256:bf77dc26e48ea95fca9d1aceb5acfa69d2e546b765ec2abfb502975f1a2d4def'

docker compose -f docker-compose.pg388.yml -p pg388demo config --quiet
docker compose -f docker-compose.pg388.yml -p pg388demo up -d --build
```

打开 [http://localhost:3000/pg388](http://localhost:3000/pg388)，并验证：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3000/pg388 | Select-Object -ExpandProperty StatusCode
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3000/pg388-api/health | Select-Object -ExpandProperty StatusCode
```

两次应返回 `200`。默认 compose 不启动 B holdout；B 只有在单独完成 image/reset/evaluator 审阅后才允许启用。

## 3. 推荐演示顺序

1. 在 `CASE MATRIX` 选择“价格 / 金额”“优惠券重放”“找回绑定”“2FA 顺序”“水平越权”或“订单并发提交”。
2. 先讲 `invariant`、`preconditions` 和 `counterfactual`，再点击本地 loopback replay。
3. 展示 `reset → observe/ASK → repair → candidate → reference → negative → replay` 轨迹；重点看 negative 是否保持 clean，以及 failure 后 action 是否改变。
4. 展开 `MODEL READINESS / STRUCTURED RULE-IR`：先看 typed/fresh/source-row 合同，再看 `CANDIDATE MODEL / CPU SMOKE` 的 ASK、最弱 head、holdout 和 false-allow。
5. 最后说明 `HOLD` 的原因：A/B 都是 implementation holdout，尚无正式 train split、operator review 和 image attestation；CPU smoke 是 wiring/结构诊断，不是“模型已经会找漏洞”。

## 4. 演示时的准确表述

可以说：

- 模型学习的是状态、不变量、角色、顺序、失败反馈和 Rule-IR 槽位；
- 本地 evaluator 能观察到受控的 typed state-shape 差分；
- candidate/reference/negative/replay 和 fresh reset 被分开审计；
- 页面同时显示模型的失败指标，而不是只展示高分。

不要说：

- 已经能攻击任意网址；
- 已经生成通用 XSS/SQL/WAF 绕过字符串；
- `confirmed_positive` 等价于真实漏洞或可迁移 payload；
- CPU smoke 或 holdout 行可以直接晋级训练。

## 5. 结束与清理

```powershell
docker compose -f docker-compose.pg388.yml -p pg388demo down
```

只清理由本次 compose 创建的容器和网络；不要使用 broad prune，也不要删除 `research/`、归档权重或受保护报告。
