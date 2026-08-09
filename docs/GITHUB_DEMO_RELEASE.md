# GitHub / A800 演示发布约定

本项目把代码、演示靶场、抽象数据和模型权重分开发布。Git 仓库是可复现的源代码入口；A800 只是训练/推理缓存，不是唯一的事实来源。

## 明天演示的最小发布集

PG-388 前端演示不依赖大模型权重即可启动：

```powershell
$env:PG388_PYTHON_IMAGE_DIGEST='sha256:<reviewed-python-digest>'
$env:PG388_NODE_BASE_IMAGE='node:20.11.1-alpine3.19@sha256:<reviewed-node-digest>'
docker compose -f docker-compose.pg388.yml -p pg388demo up -d
```

打开 `http://localhost:3000/pg388`。展示所需的抽象报告和数据集列在
`research/pg388_demo_asset_manifest_v1.json`，不会包含原始 payload、wire、响应正文、凭据或外部回调。

当前已验证的 GitHub `main` 提交是 `ebaa94dbad4da1f65864712653f837c9dc04d0d5`；截至本说明更新时，仓库还没有名为 `v0.1-demo` 的 GitHub Release。源码克隆可以立即用于前端演示，Release 下载命令只在该 Release 实际创建后使用。

## 大文件分层

| 层 | 放什么 | 接收方如何拿到 |
| --- | --- | --- |
| GitHub `main` | 源码、Docker 配置、schema、README、审计报告、manifest | `git clone` |
| GitHub Release | 明天实际展示所需的少量抽象数据集和一个候选 checkpoint | Release asset 下载后校验 SHA-256 |
| A800 缓存 | 训练时需要的工作副本、临时 checkpoint | 训练前从归档/Release 复制；训练后重新计算哈希 |
| 长期归档/对象存储 | 全量历史数据、多个 checkpoint、超过 Release 单文件限制的文件 | 通过 manifest 中的受控地址或人工发放 |

不要把唯一一份权重或数据集只放在 A800。远端 GPU 可能被回收、重装或被其他作业占用；canonical copy 必须保留在本地归档或受控对象存储中。

## 其他 AI 的接收流程

推送成功后，接收方可以按下面流程获取同一份演示材料：

```bash
git clone https://github.com/wrench1997/blackboxanalyze.git
cd blackboxanalyze
powershell -ExecutionPolicy Bypass -File scripts/verify_demo_assets.ps1 \
  -ManifestPath research/pg388_demo_asset_manifest_v1.json \
  -Root . \
  -AllowMissingOptional
```

这一步会校验 Git 中的源码/小型报告，并明确列出尚未下载的可选 checkpoint；它不会把缺失的权重伪装成已验证。若之后发布了 `v0.1-demo`，先运行：

```bash
gh release download v0.1-demo -R wrench1997/blackboxanalyze -D .
```

然后去掉 `-AllowMissingOptional` 重新校验，才可把 checkpoint 当作演示输入。

没有 GitHub CLI 时，也可以从 Release 页面逐个下载，然后运行同一个校验脚本。校验失败时不要训练或启动演示。

## A800 复制规则

A800 上的工作目录应至少保留：

```text
checkpoint.pt
dataset.json
manifest.json
SHA256SUMS
```

复制完成后先校验，训练完成后再校验 checkpoint；报告中记录 dataset、runner、vocabulary、rules 和 checkpoint 的 SHA-256。A800 上的文件不自动成为训练集、长期记忆或 payload catalog。

## 安全边界

- 只发布抽象逻辑漏洞样本和受控本地靶场代码。
- 原始攻击字符串、原始响应、网络 wire、凭据、外连地址和 evaluator 答案不进入仓库或模型上下文。
- PG-385/PG-386 的过滤反馈只能解释为本地 canary 证据，不能宣传为任意网址 WAF 绕过或通用 payload 能力。
- 任意 release asset 进入训练前，仍需通过对应 dataset/audit/rules hash gate。
